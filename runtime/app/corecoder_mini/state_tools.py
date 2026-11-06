"""Durable task drafts/Todo and explicit scoped long-term memory."""

import copy
import hashlib
import json
import re
import threading
from pathlib import Path
from typing import ClassVar

from .delivery import read_json
from .legacy_memory import Scratchpad
from .project_memory import HybridMemoryStore
from .task_state import atomic_json
from .tools.base import Tool

TASK_KINDS = frozenset({'implement', 'fix', 'refactor', 'test', 'explain', 'locate', 'review'})


class WorkingState:
    def __init__(self, directory):
        self.path = Path(directory) / "working-state.json"
        self.lock = threading.RLock()
        self.data = read_json(self.path, {"revision": 0, "notes": {}, "todos": []})

    def bind_task_contract(self, kind, source='operator'):
        """Bind declared work type, never a model guess, permission or acceptance."""
        if not isinstance(kind, str) or kind not in TASK_KINDS or not isinstance(source, str) or source not in {'operator', 'swebench_task_definition'}:
            raise ValueError('invalid_task_contract')
        contract = {'kind': kind, 'source': source}
        existing = self.data.get('task_contract')
        if existing:
            if existing != contract:
                raise ValueError('task_contract_changed_on_restart')
            return
        self.update(lambda d: d.update(task_contract=contract))

    def requested_kind(self):
        return self.data.get('task_contract', {}).get('kind') or self.data.get('intent', {}).get('kind')

    def bind_goal(self, goal):
        """Keep the exact request locally; model summaries can never replace it."""
        if not goal:
            return
        signature = hashlib.sha256(goal.encode()).hexdigest()
        if self.data.get("request_sha256"):
            if self.data["request_sha256"] != signature:
                raise ValueError("working_state_belongs_to_different_task")
            return
        from .delivery import atomic_bytes
        atomic_bytes(self.path.parent / "private/user-request.txt", goal.encode("utf-8"))
        self.update(lambda d: d.update(request_sha256=hashlib.sha256(goal.encode()).hexdigest()))

    def update(self, function, expected_revision=None):
        with self.lock:
            if expected_revision is not None and expected_revision != self.data["revision"]:
                raise ValueError("stale_working_state_revision")
            candidate = copy.deepcopy(self.data)
            function(candidate)
            candidate["revision"] += 1
            if len(json.dumps(candidate, ensure_ascii=False)) > 48000:
                raise ValueError("working_state_capacity")
            atomic_json(self.path, candidate)
            self.data = candidate
            return copy.deepcopy(candidate)

    def view(self, max_chars=4800):
        with self.lock:
            result = {"revision": self.data["revision"], "todo_is_model_declared_not_verification": True,
                      "todos": self.data["todos"], "notes": {}, "recall": "scratchpad read key"}
            if self.data.get('task_contract'):
                result['task_contract'] = self.data['task_contract']
            if self.data.get("intent"):
                intent = self.data["intent"]
                result["intent"] = {**intent, "requirements": [{**r, "behavior": r["behavior"][:160]}
                                                               for r in intent.get("requirements", [])]}
                result["intent_full_recall"] = "todo_write action=read"
            if self.data.get("request_sha256"):
                result["original_request_sha256"] = self.data["request_sha256"]
            if self.data.get("reflection"):
                result["reflection"] = self.data["reflection"]
            omitted = []
            # Current hypotheses/next steps win over old insertion order.
            priority = {"hypothesis": 0, "next_action": 1, "next_experiment": 2, "ruled_out": 3}
            for key, value in sorted(self.data["notes"].items(), key=lambda item: priority.get(item[0], 4)):
                result["notes"][key] = value
                if len(json.dumps(result, ensure_ascii=False)) > max_chars:
                    del result["notes"][key]
                    omitted.append(key)
            if len(json.dumps(result, ensure_ascii=False)) > max_chars:
                result["todos"] = [{"id": t["id"], "status": t["status"]} for t in self.data["todos"]]
                result["recall"] += "; todo_write action=read for full Todo"
            if omitted:
                result["omitted_note_count"] = len(omitted)
                result["omitted_note_keys"] = omitted[:4]
            if len(json.dumps(result, ensure_ascii=False)) > max_chars and "intent" in result:
                result["intent"] = {"kind": self.data["intent"]["kind"],
                                    "requirement_ids": [r["id"] for r in self.data["intent"]["requirements"]],
                                    "full_contract": "todo_write action=read"}
            if len(json.dumps(result, ensure_ascii=False)) > max_chars and "reflection" in result:
                result["reflection"] = {"next_action": self.data["reflection"]["next_action"][:160],
                                        "full_reflection": "todo_write action=read"}
            return result


class ScratchpadTool(Tool):
    name = "scratchpad"
    description = "Persist hypotheses and next experiments. write REQUIRES key and value, e.g. action=write,key=hypothesis,value=your note. merge requires an object; without key it updates top-level notes. read may omit key to recover all notes. delete requires key."
    parameters: ClassVar[dict] = {"type": "object", "properties": {
        "action": {"type": "string", "enum": ["read", "write", "merge", "delete"]},
        "key": {"type": "string", "maxLength": 120, "description": "Required and nonempty for write/delete. Optional for read/merge."},
        "value": {"description": "For merge, an object such as {hypothesis:...,next_experiment:...}; use write for plain text."},
        "expected_revision": {"type": "integer", "minimum": 0}, "offset": {"type": "integer", "minimum": 0},
        "limit": {"type": "integer", "minimum": 100, "maximum": 4000}}, "required": ["action"], "additionalProperties": False}

    def __init__(self, state):
        self.state = state

    def validate_arguments(self, arguments):
        action = arguments.get('action')
        if action in {'write', 'delete'} and not arguments.get('key'):
            raise ValueError('scratchpad ' + action + ' requires a nonempty key; for a note use action=write,key=hypothesis,value=your note')
        if action == 'write' and 'value' not in arguments:
            raise ValueError('scratchpad write requires value; use action=write,key=hypothesis,value=your note')
        if arguments.get("action") != "merge":
            return
        value = arguments.get("value")
        if isinstance(value, str) and len(value) <= 32000:
            try:
                value = json.loads(value)
            except ValueError:
                pass
        if not isinstance(value, dict) or not all(isinstance(k, str) and k for k in value):
            raise ValueError('scratchpad merge requires an object; use action=write with key and value for plain text')

    def execute(self, action, key="", value=None, expected_revision=None, offset=0, limit=4000):
        self.validate_arguments({'action': action, 'key': key, 'value': value})
        if action == "read":
            with self.state.lock:
                value = self.state.data["notes"].get(key) if key else self.state.data["notes"]
                encoded = json.dumps(value, ensure_ascii=False)
                return json.dumps({"revision": self.state.data["revision"], "fragment": encoded[offset:offset + limit],
                                   "offset": offset, "total_chars": len(encoded),
                                   "next_offset": offset + limit if offset + limit < len(encoded) else None}, ensure_ascii=False)

        def mutate(data):
            draft = Scratchpad()
            draft.notes = copy.deepcopy(data["notes"])
            draft.execute(action, key, value)
            data["notes"] = draft.notes

        updated = self.state.update(mutate, expected_revision)
        return json.dumps({"revision": updated["revision"], "stored_keys": list(updated["notes"])}, ensure_ascii=False)


class TodoTool(Tool):
    name = "todo_write"
    description = "Record complete intent/requirements, constraints, a bounded plan and repair reflection in one call. Check coverage against real current-workspace receipts. Done is not acceptance."
    parameters: ClassVar[dict] = {"type": "object", "properties": {
        "action": {"type": "string", "enum": ["read", "write", "check"]},
        "intent": {"type": "object", "properties": {
            "kind": {"type": "string", "enum": ["implement", "fix", "refactor", "test", "explain", "locate", "review"]},
            "summary": {"type": "string", "minLength": 1, "maxLength": 800},
            "requirements": {"type": "array", "minItems": 1, "maxItems": 12, "items": {"type": "object", "properties": {
                "id": {"type": "string", "minLength": 1, "maxLength": 40},
                "behavior": {"type": "string", "minLength": 1, "maxLength": 600}},
                "required": ["id", "behavior"], "additionalProperties": False}},
            "constraints": {"type": "array", "maxItems": 12, "items": {"type": "string", "maxLength": 300}}},
            "required": ["kind", "summary", "requirements"], "additionalProperties": False},
        "reflection": {"type": "object", "properties": {
            "failed_approach": {"type": "string", "minLength": 1, "maxLength": 600},
            "hypothesis": {"type": "string", "minLength": 1, "maxLength": 600},
            "next_action": {"type": "string", "minLength": 1, "maxLength": 600}},
            "required": ["failed_approach", "hypothesis", "next_action"], "additionalProperties": False},
        "request_offset": {"type": "integer", "minimum": 0},
        "tasks": {"type": "array", "maxItems": 12, "items": {"type": "object", "properties": {
            "id": {"type": "string", "minLength": 1, "maxLength": 40},
            "content": {"type": "string", "minLength": 1, "maxLength": 600},
            "status": {"type": "string", "enum": ["pending", "in_progress", "done", "blocked"]},
            "dependencies": {"type": "array", "maxItems": 12, "items": {"type": "string"}},
            "owner": {"type": "string", "maxLength": 40},
            "requirements": {"type": "array", "maxItems": 12, "items": {"type": "string", "maxLength": 40}},
            "evidence": {"type": "array", "maxItems": 8, "items": {"type": "string", "maxLength": 100}}},
            "required": ["content", "status"], "additionalProperties": False}},
        "expected_revision": {"type": "integer", "minimum": 0}}, "additionalProperties": False}

    def __init__(self, state, progress=None):
        self.state, self.progress = state, progress

    def execute(self, tasks=None, action="write", expected_revision=None, intent=None, reflection=None, request_offset=0):
        if action == "check":
            return json.dumps(self.check(), ensure_ascii=False)
        if action == "read":
            request = self.state.path.parent / "private/user-request.txt"
            full = request.read_text(encoding="utf-8") if request.exists() else ""
            return json.dumps({"revision": self.state.data["revision"], "tasks": self.state.data["todos"],
                               "intent": self.state.data.get("intent"), "reflection": self.state.data.get("reflection"),
                               "request_fragment": full[request_offset:request_offset + 4000], "request_chars": len(full),
                               "next_request_offset": request_offset + 4000 if request_offset + 4000 < len(full) else None}, ensure_ascii=False)
        if tasks is None and intent is None and reflection is None:
            raise ValueError("tasks_intent_or_reflection_required")
        tasks = self.state.data["todos"] if tasks is None else tasks
        checked = [{**t, "id": t.get("id", f"T{i}"), "dependencies": t.get("dependencies", []),
                    "owner": t.get("owner", "main")} for i, t in enumerate(tasks, 1)]
        by_id = {t["id"]: t for t in checked}
        if len(by_id) != len(checked) or sum(t["status"] == "in_progress" for t in checked) > 2:
            raise ValueError("duplicate_ids_or_too_many_active_todos")
        visited, active = set(), set()

        def visit(identity):
            if identity not in by_id or identity in active:
                raise ValueError("unknown_or_cyclic_todo_dependency")
            if identity in visited:
                return
            active.add(identity)
            for dep in by_id[identity]["dependencies"]:
                visit(dep)
                if by_id[identity]["status"] in {"in_progress", "done"} and by_id[dep]["status"] != "done":
                    raise ValueError("todo_dependency_not_done")
            active.remove(identity)
            visited.add(identity)

        for identity in by_id:
            visit(identity)
        contract = intent or self.state.data.get("intent", {})
        req_ids = [r["id"] for r in contract.get("requirements", [])]
        if len(req_ids) != len(set(req_ids)):
            raise ValueError("duplicate_requirement_ids")
        if any(set(t.get("requirements", [])) - set(req_ids) for t in checked):
            raise ValueError("unknown_todo_requirement")
        def mutate(data):
            data["todos"] = checked
            if intent is not None:
                data["intent"] = intent
            if reflection is not None:
                data["reflection"] = reflection
        updated = self.state.update(mutate, expected_revision)
        return json.dumps({"revision": updated["revision"], "stored_tasks": len(checked),
                           "recall": "todo_write action=read", "done_is_not_verification": True})

    def check(self):
        data = self.state.data
        requirements = data.get("intent", {}).get("requirements", [])
        rows = []
        version = self.progress.version() if self.progress else None
        for req in requirements:
            tasks = [t for t in data["todos"] if req["id"] in t.get("requirements", [])]
            receipts = []
            for task in tasks:
                for ref in task.get("evidence", []):
                    if re.fullmatch(r"action:[1-9][0-9]*", ref):
                        receipt = read_json(self.state.path.parent / "actions" / f"{int(ref[7:]):04d}.json", {})
                        if (receipt.get("status") == "executed" and receipt.get("result", {}).get("returncode") == 0
                                and version is not None and receipt.get("workspace_sha256") == version
                                and receipt.get("verification_candidate") is True):
                            receipts.append(ref)
            rows.append({"id": req["id"], "mapped_todos": [t["id"] for t in tasks],
                         "model_declared_done": bool(tasks) and all(t["status"] == "done" for t in tasks),
                         "current_successful_check_candidates": sorted(set(receipts))})
        return {"requirements": rows, "intent_recorded": bool(requirements),
                "pending_todos": [t["id"] for t in data["todos"] if t["status"] != "done"],
                "uncovered_requirements": [r["id"] for r in rows if not r["mapped_todos"]],
                "acceptance": False, "note": "Command candidates and model coverage claims need independent artifact verification."}


class ProjectMemoryTool(Tool):
    name = "memory"
    description = "Project-scoped long-term facts with local vector/BM25/keyword hybrid search. No automatic learning. Benchmark tasks cannot read/write cross-task memory. Use scratchpad for current hypotheses."
    parameters: ClassVar[dict] = {"type": "object", "properties": {
        "action": {"type": "string", "enum": ["search", "save", "read", "forget", "status", "reindex"]},
        "query": {"type": "string", "maxLength": 1000}, "text": {"type": "string", "maxLength": 4000},
        "key": {"type": "string", "maxLength": 120}, "identity": {"type": "string", "maxLength": 64},
        "limit": {"type": "integer", "minimum": 1, "maximum": 16}},
        "required": ["action"], "additionalProperties": False}

    def __init__(self, path, scope, enabled=True, settings=None):
        self.store, self.context_cache = None, {}
        if enabled:
            from .memory_embeddings import (
                OllamaMemoryEmbedder,
                validate_memory_settings,
            )
            # Direct library construction remains offline. The runtime passes its
            # explicit settings, whose validated default is local Ollama.
            options = dict(settings if settings is not None else {"memory_embedding": "off"})
            validate_memory_settings(options)
            embedder = OllamaMemoryEmbedder(options["memory_embedding_model"], options["memory_embedding_url"],
                                           options["memory_embedding_seconds"]) if options["memory_embedding"] == "ollama" else None
            self.store = HybridMemoryStore(path, scope, embedder)

    def execute(self, action, query="", text="", key="", identity="", limit=8):
        if self.store is None:
            return json.dumps({"disabled": "benchmark_cross_task_memory_quarantined", "use": "scratchpad"})
        from .runtime import check_cancelled
        check_cancelled()
        if action not in {"search", "save", "read", "forget", "status", "reindex"}:
            raise ValueError("unknown_memory_action")
        if action == "status":
            return json.dumps(self.store.status())
        if action == "reindex":
            self.context_cache.clear()
            return json.dumps(self.store.reindex(limit))
        if action == "save":
            if re.search(r"\bsk-[A-Za-z0-9_.-]{8,}|-----BEGIN .*PRIVATE KEY", text):
                raise ValueError("credential_like_memory_rejected")
            self.context_cache.clear()
            return json.dumps({"identity": self.store.save(text, kind="fact", key=key),
                               "embedding_error": self.store.embedding_error})
        if action == "search":
            return json.dumps([{**row, "text": row["text"][:600], "full_text": "memory read identity"}
                               for row in self.store.search(query, limit=4)], ensure_ascii=False)
        if action == "forget":
            self.context_cache.clear()
            return json.dumps({"forgotten": self.store.forget(identity)})
        with self.store._db() as db:
            row = db.execute("SELECT text FROM memories WHERE id=? AND scope=? AND (expires IS NULL OR expires>strftime('%s','now'))",
                             (identity, self.store.scope)).fetchone()
        if row is None:
            raise ValueError("memory_not_found_in_project_scope")
        return json.dumps({"identity": identity, "text": row[0]}, ensure_ascii=False)

    def context(self, goal):
        """One bounded startup recall; full notes stay behind explicit memory read."""
        if not self.store or not goal:
            return ""
        signature = hashlib.sha256(goal.encode()).hexdigest()
        if signature not in self.context_cache:
            rows = self.store.search(goal[:1000], limit=2)
            data = [{"identity": r["id"], "text": r["text"][:300],
                     "channels": sorted(r["channels"]), "recall": "memory read identity"} for r in rows]
            self.context_cache = {signature: json.dumps(data, ensure_ascii=False) if data else ""}
        return self.context_cache[signature]
