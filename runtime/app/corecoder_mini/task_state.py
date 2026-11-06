"""Atomic task snapshots and an append-only execution journal; no write replay."""

import copy
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path


def replace_file(temporary, path, timeout=2.0):
    """Retry only transient Windows sharing errors, using the already-fsynced file."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            os.replace(temporary, path)
            return
        except PermissionError as exc:
            if os.name != "nt" or getattr(exc, "winerror", None) not in {5, 32, 33} or time.monotonic() >= deadline:
                raise
            time.sleep(min(.025, max(0, deadline - time.monotonic())))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        replace_file(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def repair_messages(messages):
    """Reconcile each assistant/tool group, dropping orphans and filling missing replies."""
    output, index = [], 0
    while index < len(messages):
        message = copy.deepcopy(messages[index])
        index += 1
        if message.get("role") == "tool":
            continue
        calls = message.get("tool_calls") or []
        if not calls:
            output.append(message)
            continue
        ids = [call["id"] for call in calls]
        if len(set(ids)) != len(ids):
            raise ValueError("Cannot restore duplicate tool-call IDs")
        answers = {}
        while index < len(messages) and messages[index].get("role") == "tool":
            answer = messages[index]
            if answer.get("tool_call_id") in ids:
                answers[answer["tool_call_id"]] = copy.deepcopy(answer)
            index += 1
        output.append(message)
        for identity in ids:
            output.append(answers.get(identity, {"role": "tool", "tool_call_id": identity,
                "content": "[interrupted: outcome unknown; inspect current files before any retry; never replay writes blindly]"}))
    return output


class TaskStore:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.lease_token = uuid.uuid4().hex
        self.claim()
        self.sequence = 0
        self.events = []
        try:
            self.read_journal()
        except BaseException:
            self.release()
            raise

    def read_journal(self):
        journal = self.directory / "events.jsonl"
        if journal.exists():
            lines = journal.read_bytes().splitlines(keepends=True)
            valid = []
            repair = False
            for index, line in enumerate(lines):
                try:
                    event = json.loads(line)
                except (ValueError, UnicodeError):
                    if index != len(lines) - 1:
                        raise ValueError("Task journal corrupted before its final record") from None
                    # Preserve the damaged tail separately, then repair only the incomplete append.
                    (self.directory / ("damaged-tail-" + uuid.uuid4().hex + ".bin")).write_bytes(line)
                    repair = True
                    break
                if (not isinstance(event, dict) or not isinstance(event.get("sequence"), int)
                        or event["sequence"] <= self.sequence):
                    raise ValueError("Task journal sequence is invalid")
                self.sequence = event["sequence"]
                self.events.append(event)
                valid.append(line.rstrip(b"\r\n") + b"\n")
                repair |= not line.endswith(b"\n")
            if repair:
                temporary = journal.with_name("events-repair-" + uuid.uuid4().hex + ".tmp")
                with temporary.open("wb") as stream:
                    stream.write(b"".join(valid))
                    stream.flush()
                    os.fsync(stream.fileno())
                replace_file(temporary, journal)

    def claim(self):
        lease = self.directory / "lease.json"
        for _ in range(2):
            try:
                descriptor = os.open(lease, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                owner = json.loads(lease.read_text())
                pid = owner["pid"]
                if os.name == "nt":
                    import ctypes
                    handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
                    alive = bool(handle) or ctypes.windll.kernel32.GetLastError() == 5
                    if handle:
                        ctypes.windll.kernel32.CloseHandle(handle)
                else:
                    try:
                        os.kill(pid, 0)
                        alive = True
                    except ProcessLookupError:
                        alive = False
                    except PermissionError:
                        alive = True
                if alive:
                    raise RuntimeError("Task is already owned by a live process") from None
                lease.unlink()
            else:
                with os.fdopen(descriptor, "w") as stream:
                    json.dump({"pid": os.getpid(), "token": self.lease_token}, stream)
                return
        raise RuntimeError("Could not acquire task lease")

    def release(self):
        lease = self.directory / "lease.json"
        with self.lock:
            if lease.exists() and json.loads(lease.read_text()).get("token") == self.lease_token:
                lease.unlink()

    def append(self, kind, **data):
        with self.lock:
            self.sequence += 1
            event = {"sequence": self.sequence, "id": f"e{self.sequence:06d}", "time": time.time(), "type": kind, **data}
            with (self.directory / "events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            self.events.append(event)
            return event

    def save(self, state):
        with self.lock:
            state["journal_sequence"] = self.sequence
            atomic_json(self.directory / "state.json", state)
            rows = [f"# Task {state['id']}", "", state["goal"], "", f"Status: {state['status']}", ""]
            for step in state.get("steps", []):
                rows.append(f"- [{'x' if step['status'] == 'completed' else ' '}] {step['id']} "
                            f"({step['status']}, {step['owner']}) {step['title']}")
                rows.append(f"  Depends: {step['depends_on']}; needs: {step['requires']}; evidence: {step.get('evidence', [])}")
            contract = state.get("acceptance_contract", {})
            if contract.get("criteria"):
                rows.extend(["", "## Requirement checks (recorded observations; freshness is checked at completion)", ""])
                for item in contract["criteria"]:
                    rows.append(f"- {item['id']}: {item['behavior']}")
                    for phase in ("before", "after"):
                        observation = item.get(phase, {})
                        rows.append(f"  {phase}: eligible={observation.get('eligible', False)}; evidence={observation.get('event_id', 'missing')}")
                rows.append(f"Coverage review: {bool(contract.get('review'))}")
            (self.directory / "TODO.md").write_text("\n".join(rows) + "\n", encoding="utf-8")

    def load(self):
        with self.lock:
            state = json.loads((self.directory / "state.json").read_text(encoding="utf-8"))
            # A durable terminal event can outlive a failed state replacement.
            # Recover only non-success status, never infer completion or replay a tool.
            newer = [e for e in self.events if e["sequence"] > state.get("journal_sequence", 0)]
            terminal = next((e for e in reversed(newer) if e["type"] == "task_stopped"
                             and e.get("status") in {"failed", "interrupted", "incomplete"}), None)
            if (terminal and state.get("status") != terminal["status"]
                    and not any(e["sequence"] > terminal["sequence"] and e["type"] in {
                        "model_request", "official_acceptance", "task_resumed", "step_resumed"} for e in newer)):
                previous = state.get("status")
                state["status"] = terminal["status"]
                state["messages"] = repair_messages(state.get("messages", []))
                self.append("terminal_state_recovered", stopped_event=terminal["id"], previous_status=previous,
                            status=state["status"], replayed=False)
                state["journal_sequence"] = self.sequence
                atomic_json(self.directory / "state.json", state)
            return state


def safe_task_id(value):
    if not re.fullmatch(r"[a-f0-9]{32}", value):
        raise ValueError("Task ID must be 32 lowercase hexadecimal characters")
    return value
