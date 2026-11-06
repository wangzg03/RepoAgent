"""Supervisor-owned artifact export and evidence-bound completion."""

import hashlib
import json
import os
import time
import uuid
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

from scripts.swebench_io import filesystem_patch, tree_hashes

from .task_state import atomic_json


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def atomic_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def lease(directory):
    """OS lock disappears after a crash. Recovery never replays inference/actions."""
    path = Path(directory) / "run.lock"
    with path.open("a+b") as stream:
        stream.seek(0)
        if path.stat().st_size == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def export_artifact(directory, terminal):
    """No child result file is required. A missing checkpoint is itself a result."""
    directory = Path(directory)
    frozen = read_json(directory / "frozen.json")
    if tree_hashes(directory / "baseline") != frozen["baseline_sha256"]:
        raise ValueError("immutable_baseline_drift")
    patch = filesystem_patch(directory / "baseline", directory / "workspace")
    atomic_bytes(directory / "model.patch", patch)
    submission = read_json(directory / "agent/submission.json", {})
    trajectory = read_json(directory / "agent/trajectory.json", {})
    result = {"schema": 1, "task_id": frozen["task_id"], "state": "exported",
              "model_submitted": submission.get("model_submitted") is True,
              "terminal": terminal, "patch_sha256": digest(directory / "model.patch"),
              "patch_bytes": len(patch), "official_correct": None, "autonomous_complete": False,
              "checkpoint_exit": trajectory.get("info", {}).get("exit_status"),
              "usage": trajectory.get("info", {}).get("provider_usage", {}), "time": time.time()}
    requests = [read_json(p) for p in sorted((directory / "agent/requests").glob("*.json"))]
    child_requests = [read_json(p) for p in (directory / "agent/children").glob("*/agent/requests/*.json")]
    result["request_usage"] = {"main_requests": len(requests), "child_requests": len(child_requests),
                              "reported_prompt_tokens": sum(r.get("prompt_tokens", 0) for r in requests + child_requests),
                              "reported_completion_tokens": sum(r.get("completion_tokens", 0) for r in requests + child_requests),
                              "usage_complete": all(r.get("usage_complete") is True for r in requests + child_requests)}
    rows = requests + child_requests
    # Keep legacy intent counts, but identify actual transports explicitly.
    result["request_usage"].update(
        sent_requests=sum(r.get("transport_started") is True for r in rows),
        blocked_before_transport=sum(r.get("transport_started") is False for r in rows),
        transport_count_complete=all("transport_started" in r for r in rows))
    groups = {}
    for index, row in enumerate(rows):
        groups.setdefault(row.get("logical_call_id", f"legacy-{index}"), []).append(row)
    failed = [r for r in rows if r.get("diagnostic")]
    result["provider_recovery"] = {
        "logical_calls": len(groups),
        "retry_transports": sum(r.get("transport_started") is True and r.get("attempt", 1) > 1 for r in rows),
        "recovered_provider_calls": sum(any(r.get("diagnostic") for r in group) and
                                         any(r.get("status") == "returned" for r in group) for group in groups.values()),
        "errors_by_category": dict(Counter(r["diagnostic"]["category"] for r in failed)),
        "http_statuses": dict(Counter(str(r["diagnostic"].get("http_status")) for r in failed)),
        "retry_stop_reasons": dict(Counter(r["retry_decision"]["reason"] for r in failed if
                                            r.get("retry_decision") and not r["retry_decision"]["will_retry"]))}
    component_names = {key for r in rows for key in r.get("estimated_token_components", {})}
    result["token_diagnostics"] = {
        "estimated_only": True,
        "request_components": {key: sum(r.get("estimated_token_components", {}).get(key, 0) for r in rows)
                               for key in sorted(component_names)},
        "projection_estimated_tokens_saved": sum(r.get("projection", {}).get("estimated_tokens_saved", 0) for r in rows),
        "reduced_observations": sum(len(r.get("projection", {}).get("output_reductions", [])) for r in rows),
        "reported_cached_prompt_tokens": sum(r.get("cached_prompt_tokens") or 0 for r in rows),
        "cache_usage_complete": bool(rows) and all(r.get("cached_prompt_tokens") is not None for r in rows),
        "billing_note": "Configured tariff estimates are not invoices. Third rate applies only to reported cached input; unknown usage retains the full reservation."}
    from .accounting import usage_charge
    rates = frozen.get("model", {}).get("pricing_usd_per_million")
    sent = [r for r in rows if r.get("transport_started") is not False]
    charges = [usage_charge({"prompt_tokens": r.get("prompt_tokens"), "completion_tokens": r.get("completion_tokens"),
                             "prompt_cache_hit_tokens": r.get("cached_prompt_tokens")}, rates)
               if rates is not None and r.get("usage_complete") is True else None for r in sent]
    result["cost_accounting"] = {"known_usage_tariff_usd": sum(c["cost_usd"] for c in charges if c is not None),
                                  "unknown_usage_requests": sum(c is None for c in charges),
                                  "configured_cache_rate": rates[2] if rates and len(rates) == 3 else None,
                                  "cache_usage_complete": bool(charges) and all(c and c["cache_usage_complete"] for c in charges),
                                  "is_invoice": False}
    from .experience import observation
    result["harness_observation"] = observation(directory)
    progress = read_json(directory / "agent/workspace-progress.json", {})
    result["workspace_progress"] = {k: v for k, v in progress.items() if k not in {"baseline", "current", "read_receipts", "check_signatures"}}
    working = read_json(directory / "agent/working-state.json", {})
    result["task_coverage"] = {"intent_recorded": bool(working.get("intent")),
                               "requirements": len(working.get("intent", {}).get("requirements", [])),
                               "pending_todos": [t["id"] for t in working.get("todos", []) if t["status"] != "done"],
                               "reflection_recorded": bool(working.get("reflection")), "is_acceptance": False}
    atomic_json(directory / "result.json", result)
    return result


def bind_verdict(directory, verdict, evidence_path, *, kind):
    directory = Path(directory)
    result = read_json(directory / "result.json")
    if digest(directory / "model.patch") != result["patch_sha256"]:
        raise ValueError("exported_patch_drift")
    # The scorer must identify the exact bytes it actually evaluated.
    if verdict.get("patch_sha256") != result["patch_sha256"]:
        raise ValueError("scored_patch_mismatch")
    valid = verdict.get("ok") is True
    correct = verdict.get("resolved") is True and verdict.get("patch_applied") is True if valid else None
    result.update(state=("artifact_passed_incomplete" if correct else "rejected") if valid else "unscored", verification_kind=kind,
                  verification_ok=valid, artifact_correct=correct,
                  official_correct=correct if kind in {"swebench_official", "humaneval_official"} else None,
                  evidence_path=str(Path(evidence_path).resolve()), evidence_sha256=digest(evidence_path))
    result["autonomous_complete"] = bool(valid and correct and result["model_submitted"])
    if result["autonomous_complete"]:
        result["state"] = "completed"
    atomic_json(directory / "result.json", result)
    return result
