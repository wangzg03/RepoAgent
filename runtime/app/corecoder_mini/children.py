"""Bounded, read-only child investigations; no recursive agent tool or project Git."""

import copy
import json
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import ClassVar

from scripts.swebench_io import tree_hashes

from .budget import Budget, BudgetExceeded
from .delivery import read_json
from .runtime import check_cancelled, local
from .task_state import atomic_json
from .tools.base import Tool
from .upstream import Submitted


class BudgetGate:
    """A shared ledger plus explicit remaining capacity reserved for main requests."""
    def __init__(self, shared, *, main_reserve=2, child_limit=None, parent=None):
        self.shared = shared
        self.main_reserve = main_reserve
        self.child_limit = child_limit
        self.parent = parent
        self.admission_lock = parent.admission_lock if parent else threading.RLock()
        self.local_requests = 0
        self.main_quote = None

    def __getattr__(self, name):
        return getattr(self.shared, name)

    def fork(self, limit):
        return BudgetGate(self.shared, main_reserve=self.main_reserve, child_limit=limit, parent=self)

    def reserve(self, params, rates):
        with self.admission_lock:
            quote = self.shared.quote(params, rates)
            if self.parent:
                if self.local_requests >= self.child_limit:
                    raise BudgetExceeded("child_request_limit")
                available = self.shared.admission(params, rates)
                reserve = self.parent.main_quote or quote
                if ((available["available_requests"] is not None and
                     available["available_requests"] - 1 < self.main_reserve) or
                    (available["available_tokens"] is not None and
                     available["available_tokens"] - quote.tokens < reserve.tokens * self.main_reserve) or
                    available["available_usd"] - quote < float(reserve) * self.main_reserve - 1e-12):
                    raise BudgetExceeded("child_would_consume_main_reserve")
            else:
                self.main_quote = quote
            reservation = self.shared.reserve(params, rates)
            self.local_requests += 1
            return reservation


class LinkedCancel:
    def __init__(self, parent):
        self.parent, self.own = parent, threading.Event()

    def set(self):
        self.own.set()

    def is_set(self):
        return self.own.is_set() or bool(self.parent and self.parent.is_set())

    def wait(self, seconds):
        end = time.monotonic() + seconds
        while not self.is_set() and time.monotonic() < end:
            self.own.wait(min(.05, max(0, end - time.monotonic())))
        return self.is_set()


class ChildReportTool(Tool):
    name = "subagent_report"
    description = "Finish this read-only investigation with bounded findings and actual action receipt numbers. Reports are claims for main review, not main verification."
    parameters: ClassVar[dict] = {"type": "object", "properties": {
        "summary": {"type": "string", "minLength": 1, "maxLength": 1200},
        "findings": {"type": "array", "maxItems": 4, "items": {"type": "object", "properties": {
            "claim": {"type": "string", "maxLength": 500},
            "action": {"type": "integer", "minimum": 1}}, "required": ["claim", "action"], "additionalProperties": False}},
        "unresolved": {"type": "string", "maxLength": 800}},
        "required": ["summary", "findings"], "additionalProperties": False}

    def __init__(self, directory):
        self.directory = Path(directory)

    def execute(self, summary, findings, unresolved=""):
        for finding in findings:
            receipt = read_json(self.directory / "actions" / f"{finding['action']:04d}.json", {})
            if receipt.get("status") != "executed" or receipt.get("result", {}).get("returncode") != 0:
                raise ValueError("child_finding_requires_actual_successful_action")
        report = {"summary": summary, "findings": findings, "unresolved": unresolved}
        atomic_json(self.directory / "report.json", report)
        raise Submitted({"role": "exit", "content": "Child report ready for parent review.",
                         "extra": {"exit_status": "ChildReported", "submission": "", "report": report}})


class ChildManager:
    def __init__(self, llm, workspace, directory, image, settings, state, events, *, provider_factory=None):
        self.llm, self.workspace = llm, Path(workspace)
        self.directory = Path(directory)
        self.image, self.settings, self.state, self.events = image, settings, state, events
        self.provider_factory = provider_factory
        self.lock = threading.RLock()
        self.records = read_json(self.directory / "children.json", {})
        self.goal_provider = lambda: ""
        shared = getattr(llm, "budget", None) or Budget(settings["dollar_limit"], settings["token_limit"], settings["request_limit"])
        self.gate = shared if isinstance(shared, BudgetGate) else BudgetGate(shared, main_reserve=settings.get("main_request_reserve", 2))
        llm.budget = self.gate

    def save(self):
        atomic_json(self.directory / "children.json", self.records)

    def dispatch(self, jobs):
        check_cancelled()
        limit = min(2, self.settings.get("max_subagents", 2))
        if not jobs or len(jobs) > limit:
            raise ValueError("child_batch_limit")
        if any(not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", j["id"]) for j in jobs):
            raise ValueError("safe_child_ids_required")
        with self.lock:
            if len(self.records) + len(jobs) > limit:
                raise ValueError("child_lifetime_limit")
            ids = {j["id"] for j in jobs}
            if len(ids) != len(jobs) or ids & set(self.records):
                raise ValueError("duplicate_child_id")
            # Batches are independent; chains use previously delivered, reviewed reports.
            packets = []
            for job in jobs:
                permissions = {}
                if self.settings.get('repoagent'):
                    from repoagent.child_permissions import assignment
                    permissions = assignment(job, self.settings)
                accepted = []
                for dep in job.get("dependencies", []):
                    record = self.records.get(dep, {})
                    if (record.get("review") or {}).get("disposition") != "used":
                        raise ValueError("child_dependency_requires_parent_acceptance")
                    accepted.append({"id": dep, "report": record["report"]})
                goal = self.goal_provider()
                requested = job.get("requirements", [])
                contract = self.state.data.get("intent", {})
                known = {r["id"]: r for r in contract.get("requirements", [])}
                if set(requested) - known.keys():
                    raise ValueError("unknown_child_requirement")
                packets.append({"id": job["id"], "assignment": job["task"], "overall_goal": goal[:2400],
                                "kind": job.get("kind", "source"),
                                "overall_goal_truncated": len(goal) > 2400, "dependencies": accepted,
                                "original_request_recall": "trajectory_read action=request offset=0 limit=4000",
                                "requirements": [known[r] for r in requested],
                                "expected_output": job.get("expected_output", "Relevant source locations, discriminating checks and unresolved risks."),
                                "working_memory": self.state.view(2000), "write_scope": "none",
                                "output_contract": "Use subagent_report with actual action numbers; do not delegate."})
                packets[-1].update(permissions)
                if permissions.get('write_scope'):
                    packets[-1]['output_contract'] += ' Make only granted scoped changes in your isolated snapshot.'
                if job.get("kind") == "trajectory_review":
                    index = read_json(self.directory / "private/recreate-episode/trajectory_index.json", {})
                    progress = read_json(self.directory / "workspace-progress.json", {})
                    packets[-1]["episode_evidence"] = {"errors": index.get("errors", [])[-5:],
                                                       "tests": index.get("tests", {}),
                                                       "repair_windows": progress.get("recovery", {}).get("windows", [])[-2:]}
                    observations = []
                    for failed in progress.get("unresolved_commands", [])[-3:]:
                        receipt = read_json(self.directory / "actions" / f"{failed['action_id']:04d}.json", {})
                        if receipt.get("status") != "executed":
                            continue
                        output = receipt.get("result", {}).get("output", "")
                        observations.append({"parent_action": failed["action_id"], "returncode": failed["returncode"],
                            "command": str(receipt.get("action", {}).get("arguments", {}).get("command",
                                           receipt.get("action", {}).get("command", "")))[:600],
                            "output_excerpt": output[:600] + ("\n[excerpt omitted]\n" + output[-400:] if len(output) > 1000 else output[600:]),
                            "excerpt_truncated": len(output) > 1000})
                    packets[-1]["episode_evidence"]["failure_observations"] = observations
                    packets[-1]["output_contract"] += " Explain the failure mechanism, evidence, and one different executable experiment for main. Do not change the harness, limits or evaluator."
            for packet in packets:
                self.records[packet["id"]] = {"status": "queued", "input": packet, "review": None}
            self.save()
        context = dict(vars(local))
        with ThreadPoolExecutor(max_workers=min(limit, self.settings.get("child_concurrency", 2))) as pool:
            futures = [pool.submit(self._run, packet, context) for packet in packets]
            return [future.result() for future in futures]

    def _run(self, packet, inherited):
        from .engine import build_agent
        identity = packet["id"]
        directory = self.directory / "children" / identity
        snapshot = directory / "workspace"
        record = self.records[identity]
        vars(local).update(inherited)
        local.cancel = LinkedCancel(inherited.get("cancel"))
        limit = time.monotonic() + self.settings.get("child_seconds", 120)
        local.deadline = min(limit, inherited["deadline"]) if inherited.get("deadline") else limit
        agent = None
        try:
            check_cancelled()
            shutil.copytree(self.workspace, snapshot, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"))
            baseline = tree_hashes(snapshot)
            record["baseline_sha256"] = baseline
            atomic_json(directory / "input.json", packet)
            record["status"] = "running"
            self.events.emit("child_started", child_id=identity)
            budget = self.gate.fork(self.settings.get("child_requests", 3))
            if self.provider_factory:
                llm = self.provider_factory(identity, budget)
            else:
                llm = copy.copy(self.llm)
                llm.extra = copy.deepcopy(self.llm.extra)
                llm.total_prompt_tokens = llm.total_completion_tokens = 0
                llm.budget = budget
            # A provider factory cannot accidentally detach the child from the shared ledger.
            llm.budget = budget
            child_settings = {**self.settings, "step_limit": self.settings.get("child_requests", 3),
                              "tool_limit": 8, "max_subagents": 0,
                              "harness_tools": ["trajectory_read"],
                              "tool_extensions": [], "context_tokens": min(16384, self.settings["context_tokens"]),
                              "max_output_tokens": min(1536, self.settings["max_output_tokens"]),
                              "history_target_tokens": 6000, "task_kind": "benchmark"}
            if self.settings.get('repoagent'):
                child_settings.update(active_child_capabilities=packet.get('capabilities', ['read', 'test']),
                                      child_can_write='write' in packet.get('capabilities', []),
                                      write_scope=packet.get('write_scope') or ['.'],
                                      require_task_contract=False, mcp_servers={})
            if "thinking_budget" in child_settings:
                child_settings["thinking_budget"] = min(child_settings["thinking_budget"],
                                                        max(1, child_settings["max_output_tokens"] // 2))
            agent = build_agent(llm, snapshot, directory / "agent", self.image, child_settings,
                                job_id=self.settings.get("job_id", "child") + "-" + identity, depth=1)
            agent.working_state.bind_goal(self.goal_provider())
            result = agent.run(json.dumps(packet, ensure_ascii=False))
            report = result.get("report", {"summary": "Child did not deliver a structured report.",
                                          "findings": [], "unresolved": result.get("exit_status", "unknown")})
            after = tree_hashes(snapshot)
            record["changed_paths"] = sorted(p for p in set(after) | set(baseline) if after.get(p) != baseline.get(p))
            if self.settings.get('repoagent'):
                from repoagent.child_permissions import validate_changes
                validate_changes(record, record['changed_paths'])
            elif record["changed_paths"]:
                raise ValueError("read_only_child_snapshot_changed")
            record['after_sha256'] = after
            actions = sorted((directory / "agent/actions").glob("*.json"))
            record.update(status="completed" if result.get("exit_status") == "ChildReported" else "partial",
                          report=report, changed_paths=record["changed_paths"], tool_receipts=[str(p.relative_to(self.directory)) for p in actions],
                          actual_tools=[read_json(p).get("tool", "bash") for p in actions],
                          model_usage=agent.model.serialize()["info"]["provider_usage"])
        except Exception as exc:  # noqa: BLE001 - one child failure must return to the main agent
            record.update(status="failed", report={"summary": "Child investigation failed.", "findings": [],
                                                  "unresolved": type(exc).__name__}, changed_paths=record.get("changed_paths", []))
        finally:
            if agent:
                try:
                    agent.env.cleanup()
                except Exception as exc:  # noqa: BLE001 - retain report and let supervisor retry owned cleanup
                    record["cleanup_error_type"] = type(exc).__name__
            actions = sorted((directory / "agent/actions").glob("*.json"))
            record["tool_receipts"] = [str(p.relative_to(self.directory)) for p in actions]
            record["actual_tools"] = [read_json(p).get("tool", "bash") for p in actions]
            with self.lock:
                self.save()
            self.events.emit("child_finished", child_id=identity, status=record["status"])
            vars(local).clear()
        return {"id": identity, **{k: record.get(k) for k in ("status", "report", "actual_tools", "changed_paths", "tool_receipts")}}

    def review(self, identity, disposition, reason):
        with self.lock:
            record = self.records.get(identity, {})
            if record.get("status") not in {"completed", "partial", "failed"}:
                raise ValueError("child_report_not_delivered")
            findings = record.get("report", {}).get("findings", [])
            if disposition == "used" and (record.get("status") == "failed" or not findings or (record.get("changed_paths") and not self.settings.get("repoagent"))):
                raise ValueError("used_child_requires_delivered_evidence")
            record["review"] = {"disposition": disposition, "reason": reason,
                                "source_actions": sorted({f["action"] for f in findings}) if disposition == "used" else []}
            self.save()
            self.events.emit("child_reviewed", child_id=identity, disposition=disposition)
            return {"id": identity, "review": record["review"], "note": "Parent must independently verify its changed workspace."}


class SubagentTool(Tool):
    name = "subagent"
    description = "Delegate at most two nonrecursive children with isolated context and shared budgets. Request read/test or operator-authorized scoped write capabilities; review reports and explicitly integrate isolated edits, then test main."
    parameters: ClassVar[dict] = {"type": "object", "properties": {
        "action": {"type": "string", "enum": ["spawn", "review", "status", "integrate"]},
        "jobs": {"type": "array", "minItems": 1, "maxItems": 2, "items": {"type": "object", "properties": {
            "id": {"type": "string", "minLength": 1, "maxLength": 32},
            "task": {"type": "string", "minLength": 1, "maxLength": 2400},
            "capabilities": {"type": "array", "items": {"type": "string", "enum": ["read", "test", "write"]}},
            "write_scope": {"type": "array", "items": {"type": "string", "maxLength": 512}},
            "kind": {"type": "string", "enum": ["source", "tests", "trajectory_review"]},
            "requirements": {"type": "array", "maxItems": 12, "items": {"type": "string", "maxLength": 40}},
            "expected_output": {"type": "string", "maxLength": 600},
            "dependencies": {"type": "array", "maxItems": 2, "items": {"type": "string"}}},
            "required": ["id", "task"], "additionalProperties": False}},
        "identity": {"type": "string", "maxLength": 32},
        "disposition": {"type": "string", "enum": ["used", "rejected"]},
        "reason": {"type": "string", "maxLength": 1200}}, "required": ["action"], "additionalProperties": False}

    def __init__(self, manager):
        self.manager = manager

    def execute(self, action, jobs=None, identity="", disposition="used", reason=""):
        if action == "spawn":
            if not jobs or any(not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", j["id"]) for j in jobs):
                raise ValueError("safe_child_ids_required")
            return json.dumps(self.manager.dispatch(jobs), ensure_ascii=False)
        if action == 'integrate':
            if not self.manager.settings.get('repoagent'):
                raise ValueError('child_integration_not_enabled')
            from repoagent.child_permissions import integrate
            return json.dumps(integrate(self.manager, identity))
        if action == "review":
            if not reason.strip():
                raise ValueError("review_reason_required")
            return json.dumps(self.manager.review(identity, disposition, reason), ensure_ascii=False)
        return json.dumps([{ "id": k, "status": v["status"], "review": v.get("review")}
                           for k, v in self.manager.records.items()], ensure_ascii=False)
