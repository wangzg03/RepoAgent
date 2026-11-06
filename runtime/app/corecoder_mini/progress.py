"""Workspace observations, not a phase controller or an acceptance oracle."""

import hashlib
import json
import shlex
from pathlib import Path

from scripts.swebench_io import tree_hashes

from .command_observation import (
    check_command,
    command_identity,
    command_targets,
    observation_kind,
    search_result_kind,
    unavailable_git_history,
)
from .delivery import read_json
from .task_state import atomic_json


class ProgressStalled(RuntimeError):
    """No productive transition after an explicit, persisted repair allowance."""


def repository_read(command):
    """Conservative pure repository reads only. Unknown shell commands always execute."""
    if any(token in command for token in (";", "|", "&", ">", "<", "`", "$", "\n", "\\")):
        return False
    try:
        args = shlex.split(command)
    except ValueError:
        return False
    if not args or any(a.startswith("/") or ".." in a.split("/") for a in args[1:]):
        return False
    # Avoid command languages (sed 'e'), directory/ignored-state listings and
    # searches that can depend on generated files outside the source fingerprint.
    return args[0] in {"cat", "head", "tail", "wc"}


class WorkspaceProgress:
    def __init__(self, workspace, directory, settings=None, state=None):
        self.workspace = Path(workspace)
        self.directory, self.settings, self.state = Path(directory), settings or {}, state
        self.path = Path(directory) / "workspace-progress.json"
        self.data = read_json(self.path)
        if self.data is None:
            hashes = tree_hashes(self.workspace)
            self.data = {"baseline": hashes, "current": hashes, "observations": 0,
                         "actions_without_workspace_change": 0, "changed_paths": [],
                         "unresolved_commands": [], "observation_error": None}
            atomic_json(self.path, self.data)
        self.data.setdefault("recovery", {"requests": 0, "idle_requests": 0, "windows": [], "anchor": None})
        self.data.setdefault("read_receipts", {})
        self.data.setdefault("check_signatures", [])
        self.data.setdefault("productive_epoch", 0)
        self.data.setdefault("evidence_epoch", 0)
        self.data.setdefault("evidence_signatures", [])
        self.data.setdefault("recent_actions", [])
        self.data.setdefault("diagnostic_failures", [])

    def version(self):
        return hashlib.sha256(json.dumps(tree_hashes(self.workspace), sort_keys=True).encode()).hexdigest()

    def reused_read(self, name, arguments):
        if name != "bash" or not repository_read(arguments["command"]):
            return None
        hashes = tree_hashes(self.workspace)
        files = [p for p in shlex.split(arguments["command"])[1:] if p in hashes]
        if not files:
            return None
        for operand in shlex.split(arguments["command"])[1:]:
            path = self.workspace / operand
            if not operand.startswith("-") and path.exists() and (not path.is_file() or operand not in hashes):
                return None
        key = hashlib.sha256((self.version() + "\0" + arguments["command"]).encode()).hexdigest()
        action_id = self.data["read_receipts"].get(key)
        if not action_id:
            return None
        receipt = read_json(self.directory / "actions" / f"{action_id:04d}.json", {})
        if receipt.get("status") != "executed" or receipt.get("result", {}).get("returncode") != 0:
            return None
        return {**receipt["result"], "reused_from": action_id, "execution_kind": "reused_observation",
                "output": f"[Same repository version; reused action {action_id}, command NOT rerun.]\n" + receipt["result"]["output"]}

    def request_feedback(self):
        recovery = self.data["recovery"]
        recovery["requests"] += 1
        anchor = self.data["productive_epoch"]
        if recovery["anchor"] != anchor:
            recovery.update(anchor=anchor, idle_requests=0, active_window=None, no_information_requests=0)
        recovery["idle_requests"] += 1
        evidence = self.data["evidence_epoch"]
        if recovery.get("evidence_anchor") != evidence:
            recovery.update(evidence_anchor=evidence, no_information_requests=0)
        else:
            recovery["no_information_requests"] = recovery.get("no_information_requests", 0) + 1
        limit = self.settings.get("stall_request_limit", 6)
        window = self.settings.get("repair_window_requests", 3)
        kind = self.state.requested_kind() if self.state else None
        read_only = kind in {"explain", "locate", "review"}
        feedback = None
        if limit and recovery["idle_requests"] >= limit:
            active = recovery.get("active_window")
            if active is None:
                active = {"request": recovery["requests"], "anchor": anchor, "phase": "diagnose",
                          "diagnosis": "command_failures" if self.data["unresolved_commands"] else "exploration_stall",
                          "failed_actions": [r["action_id"] for r in self.data["unresolved_commands"]]}
                recovery["active_window"] = active
                recovery["windows"].append(active)
            active.setdefault("phase", "diagnose")
            # The first allowance diagnoses. Repeating the same reminder is not a
            # recovery strategy: rebuild a request view from actual receipts once.
            if active["phase"] == "diagnose" and recovery["idle_requests"] >= limit + window:
                used = recovery.get("reanchors", [])
                if len(used) < self.settings.get("max_repair_windows", 2):
                    packet = {"id": len(used) + 1, "request": recovery["requests"],
                              "anchor": anchor, "actions": list(self.data["recent_actions"][-10:]),
                              "failed_actions": [r["action_id"] for r in self.data["unresolved_commands"]]}
                    recovery["reanchors"] = used + [packet]
                    recovery["reanchor_id"] = packet["id"]
                    active.update(phase="reanchor", repair_until=recovery["requests"] + window)
                elif self._blocking_failures() or recovery["no_information_requests"] >= window:
                    self._stop()
                else:
                    active["phase"] = "converge"
            if active["phase"] in {"reanchor", "converge"}:
                expired = recovery["requests"] >= active.get("repair_until", recovery["requests"])
                if expired and (self._blocking_failures() or recovery["no_information_requests"] >= window):
                    self._stop()
                if expired:
                    active["phase"] = "converge"
            feedback = {**active, "idle_requests": recovery["idle_requests"], "read_only_intent": read_only,
                        "no_information_requests": recovery["no_information_requests"],
                        "observation_rule": "Distinct executed commands are observations, including Python probes. "
                        "They are not edits or correctness checks. Only repeated observations/bookkeeping "
                        "or unresolved execution failures can exhaust this repair window.",
                        "remaining_repair_requests": max(0, active.get("repair_until", active["request"] + window) - recovery["requests"]),
                        "next_action": self._next_action(read_only),
                        "required_next_step": "Act on the captured observations, not another broad repository survey. "
                        "Once a mismatch is located, make the smallest supported source change, run a relevant "
                        "assertion/test, repair actual failures and submit when ready. If one fact is missing, "
                        "name that uncertainty and inspect only its target. A print-only probe is diagnostic, "
                        "not verification. All tools remain available; there is no pre-edit protocol gate."}
        if recovery.get("reanchor_id"):
            feedback = {**(feedback or {}), "reanchor_id": recovery["reanchor_id"],
                        "recovery_packet": recovery["reanchors"][-1], "budgets_unchanged": True}
        atomic_json(self.path, self.data)
        return feedback

    def _stop(self):
        self.data["recovery"]["stop_reason"] = "progress_stalled_after_repair_window"
        self.data["recovery"]["stop_cause"] = (
            "unresolved_execution_failure" if self._blocking_failures() else "repeated_observations_or_bookkeeping")
        atomic_json(self.path, self.data)
        raise ProgressStalled("progress_stalled_after_repair_window; recovery attempted; saved workspace and receipts")

    def _next_action(self, read_only):
        """Advisory delivery state from actual receipts, never an acceptance gate."""
        if read_only:
            return "Deliver the supported findings and submit; do not change source for read-only work."
        if self._blocking_failures():
            if any(r.get('kind', 'check') == 'check' for r in self._blocking_failures()):
                return "Resolve the recorded failed check against the current source and rerun that check."
            return "Repair or replace the failing tool invocation using its error; do not require a search to produce a match."
        if self.search_feedback():
            return self.search_feedback()['instruction']
        if not self.data["changed_paths"]:
            return "Implementation is still missing: use the located behavior to make a scoped edit, or identify one concrete blocker."
        version = hashlib.sha256(json.dumps(self.data["current"], sort_keys=True).encode()).hexdigest()
        checked = any(r.get("check_candidate") and r["returncode"] == 0 and r.get("workspace_sha256") == version
                      for r in self.data["recent_actions"])
        if checked:
            return "Review the requested edge cases and diff; if covered, submit with echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT."
        return "Source is changed: run the relevant check against this version, repair any failure, then submit."

    def search_feedback(self):
        """A no-match is evidence, not an unsatisfied test that must turn green."""
        if self.data['changed_paths']:
            return None
        recent = self.data['recent_actions']
        if not recent or recent[-1].get('kind') != 'search':
            return None
        last = recent[-1]
        if last['returncode'] == 0:
            return None
        repeats = sum(r.get('command_sha256') == last.get('command_sha256')
                      and r.get('workspace_sha256') == last.get('workspace_sha256')
                      and r['returncode'] == last['returncode'] for r in recent)
        no_match = last.get('diagnostic_kind') == 'search_no_match'
        instruction = ('The search found no match; this is not a failed test. Do not rerun the same query. '
                       'Use a different exact symbol/path or read the already located source with read_file. '
                       'Make the supported edit once located; if a fact is missing, name that fact and inspect only its target.') if no_match else (
                       'The source search itself failed. Correct its arguments or use read_file/a supported focused search. '
                       'Do not repeat the broken invocation or treat it as a failing repository test; then continue the requested change.')
        return {'no_match': no_match, 'repetitions': repeats, 'action_id': last['action_id'],
                'target_paths': last.get('paths', [])[:2], 'instruction': instruction}

    def _blocking_failures(self):
        # A real check failure remains unresolved until the same check succeeds.
        # Repeated unknown invocations block their observed source version.
        # Keep their history after actual edits, but do not demand that an obsolete
        # diagnostic succeed. Real failed checks remain blocking across versions.
        version = hashlib.sha256(json.dumps(self.data['current'], sort_keys=True).encode()).hexdigest()
        return [row for row in self.data["unresolved_commands"]
                if row.get("kind", "check") == "check" or row.get("failure_count", 1) > 1
                and row.get('workspace_sha256', version) == version]

    def observe(self, action_id, name, arguments, result):
        self.data["observations"] += 1
        try:
            current = tree_hashes(self.workspace)
        except (OSError, ValueError) as exc:
            # Monitoring cannot falsely declare progress or disable editing.
            self.data["observation_error"] = type(exc).__name__
        else:
            self.data["observation_error"] = None
            changed = current != self.data["current"]
            if changed:
                self.data["productive_epoch"] += 1
            self.data["actions_without_workspace_change"] = (
                0 if changed else self.data["actions_without_workspace_change"] + 1)
            self.data["current"] = current
            baseline = self.data["baseline"]
            self.data["changed_paths"] = sorted(p for p in baseline.keys() | current.keys()
                                                if baseline.get(p) != current.get(p))
        annotation = {"workspace_changed": bool(locals().get("changed", False)),
                      "verification_candidate": False}
        if self.data["observation_error"] is None:
            annotation["workspace_sha256"] = self.version()
        if name in {"bash", "run_tests"}:
            kind = observation_kind(arguments["command"], self.workspace)
            if result["returncode"] != 0 and unavailable_git_history(arguments["command"], result, self.workspace):
                kind = "inspection"
            targets = command_targets(arguments["command"], self.workspace)
            annotation.update(observation_kind=kind, source_targets=targets)
            command_hash = command_identity(arguments["command"], self.workspace)
            annotation['command_sha256'] = command_hash
            previous = next((r for r in self.data["unresolved_commands"] if r["command_sha256"] == command_hash), {})
            unresolved = [r for r in self.data["unresolved_commands"] if r["command_sha256"] != command_hash]
            if result["returncode"] != 0 and kind not in {"inspection", "search"}:
                unresolved.append({"action_id": action_id, "returncode": result["returncode"],
                                   "command_sha256": command_hash, "kind": kind,
                                   "workspace_sha256": annotation.get('workspace_sha256'),
                                   "failure_count": previous.get("failure_count", 0) + 1})
            self.data["unresolved_commands"] = unresolved
            if result["returncode"] != 0 and kind in {"inspection", "search"}:
                diagnostic = search_result_kind(arguments['command'], result['returncode'], self.workspace) if kind == 'search' else 'inspection_failed'
                if "not a git repository" in result.get("output", "").lower():
                    diagnostic = "git_history_unavailable_use_source_files"
                self.data["diagnostic_failures"] = (self.data["diagnostic_failures"] + [
                    {"action_id": action_id, "kind": diagnostic, "returncode": result["returncode"]}])[-5:]
                annotation["diagnostic_kind"] = diagnostic
            # Observation novelty is independent of our shell classifier. Python
            # diagnostics and unknown-but-executed commands can reveal new facts.
            # Counting only cat/grep/etc. incorrectly killed distinct real probes.
            # This is not verification and cannot renew a failed-check allowance.
            if self.data["observation_error"] is None:
                signature = annotation.get("workspace_sha256", "unknown") + command_hash + str(result["returncode"])
                if signature not in self.data["evidence_signatures"]:
                    self.data["evidence_signatures"].append(signature)
                    self.data["evidence_epoch"] += 1
                    annotation["new_observation_evidence"] = True
                    if kind in {"inspection", "search"}:
                        annotation["new_inspection_evidence"] = True
            if repository_read(arguments["command"]) and result["returncode"] == 0 and not result.get("reused_from"):
                key = hashlib.sha256((annotation.get("workspace_sha256", "unknown") + "\0" + arguments["command"]).encode()).hexdigest()
                self.data["read_receipts"][key] = action_id
                # Session-local optimization only; old versions cannot grow the index forever.
                self.data["read_receipts"] = dict(list(self.data["read_receipts"].items())[-64:])
            if (check_command(arguments["command"], self.workspace) and not result.get("reused_from")
                    and self.data["observation_error"] is None):
                annotation["verification_candidate"] = True
                signature = hashlib.sha256((annotation.get("workspace_sha256", "unknown") + command_hash +
                                            str(result["returncode"])).encode()).hexdigest()
                if signature not in self.data["check_signatures"]:
                    self.data["check_signatures"] = (self.data["check_signatures"] + [signature])[-64:]
                    self.data["productive_epoch"] += 1
        elif name == 'code_search':
            annotation.update(observation_kind='search', source_targets=[])
            if result['returncode'] == 0:
                self.data['evidence_epoch'] += 1
                annotation['new_inspection_evidence'] = True
        elif name in {"read_file", "edit_file", "write_file"}:
            from .tools.workspace_io import validate_path
            path = '/'.join(validate_path(arguments['file_path']))
            kind = 'inspection' if name == 'read_file' else 'edit'
            annotation.update(observation_kind=kind, source_targets=[path])
            identity = hashlib.sha256(json.dumps({'tool': name, 'arguments': {**arguments, 'file_path': path}},
                                                sort_keys=True).encode()).hexdigest()
            previous = next((r for r in self.data['unresolved_commands'] if r['command_sha256'] == identity), {})
            if kind == 'edit':
                unresolved = [r for r in self.data['unresolved_commands'] if r['command_sha256'] != identity
                              and not (result['returncode'] == 0 and r.get('kind') == 'edit' and r.get('file_path') == path)]
                if result['returncode'] != 0:
                    unresolved.append({'action_id': action_id, 'returncode': result['returncode'],
                                       'command_sha256': identity, 'tool': name, 'file_path': path,
                                       'workspace_sha256': annotation.get('workspace_sha256'),
                                       'kind': 'edit', 'failure_count': previous.get('failure_count', 0) + 1})
                self.data['unresolved_commands'] = unresolved
            elif result['returncode'] != 0:
                annotation['diagnostic_kind'] = 'inspection_failed'
                self.data['diagnostic_failures'] = (self.data['diagnostic_failures'] + [
                    {'action_id': action_id, 'kind': 'inspection_failed', 'returncode': result['returncode']}])[-5:]
            if self.data['observation_error'] is None:
                signature = annotation['workspace_sha256'] + identity + str(result['returncode'])
                if signature not in self.data['evidence_signatures']:
                    self.data['evidence_signatures'].append(signature)
                    self.data['evidence_epoch'] += 1
                    annotation['new_observation_evidence'] = True
                    if kind == 'inspection':
                        annotation['new_inspection_evidence'] = True
        self.data["recent_actions"] = (self.data["recent_actions"] + [
            {"action_id": action_id, "tool": name, "returncode": result["returncode"],
             "kind": annotation.get("observation_kind", "task_metadata"),
             "paths": annotation.get("source_targets", []), "workspace_changed": annotation["workspace_changed"],
             "workspace_sha256": annotation.get("workspace_sha256"),
             "new_observation_evidence": annotation.get("new_observation_evidence", False),
             "command_sha256": annotation.get('command_sha256'),
             "diagnostic_kind": annotation.get('diagnostic_kind'),
             "check_candidate": annotation["verification_candidate"]}])[-16:]
        atomic_json(self.path, self.data)
        return annotation

    def feedback(self):
        data = self.data
        if not data["observations"]:
            return None
        paths = data["changed_paths"]
        result = {"workspace_changed_files": len(paths), "changed_paths": paths[:12],
                  "actions_without_workspace_change": data["actions_without_workspace_change"],
                  "recent_nonzero_commands": data["unresolved_commands"][-5:],
                  "unresolved_command_count": len(data["unresolved_commands"]),
                  "unresolved_check_actions": [r["action_id"] for r in self._blocking_failures()],
                  "diagnostic_observations": data["diagnostic_failures"],
                  "observation_error": data["observation_error"],
                  "meaning": "File changes are not proof of repair; nonzero commands are not necessarily tests."}
        if data["actions_without_workspace_change"] >= 6:
            result["next_action_guidance"] = (
                "No workspace change across these actions. If implementation is requested, state the current "
                "hypothesis and use edit_file/write_file for a scoped implementation or a discriminating executable check. Explain what "
                "specific uncertainty any further read resolves; reuse prior observations. For a read-only "
                "task, conclude with supported findings. No tool is disabled and no new budget is granted.")
        return result
