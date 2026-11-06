"""ReCreate-inspired experience receipts, without runtime scaffold mutation."""

import hashlib
import json
from collections import Counter
from pathlib import Path

from .delivery import digest, read_json
from .task_state import atomic_json


def observation(directory):
    directory = Path(directory)
    actions = [read_json(p) for p in sorted((directory / "agent/actions").glob("*.json"))]
    children = read_json(directory / "agent/children.json", {})
    return {"actual_tool_counts": dict(Counter(a.get("tool", "bash") for a in actions if a.get("status") in {"executed", "submitted"})),
            "tool_request_counts": dict(Counter(a.get("tool", "bash") for a in actions)),
            "reused_read_observations": sum(a.get("status") == "reused" for a in actions),
            "public_submission_rejections": sum(a.get("status") == "submission_rejected" for a in actions),
            "tool_failure_counts": dict(Counter(a.get("result", {}).get("exception_info") or "nonzero_exit"
                for a in actions if a.get("status") == "executed" and a.get("result", {}).get("returncode") not in {0, None})),
            "unresolved_write_intents": sum(a.get("status") in {"intent", "interrupted_or_failed"} for a in actions),
            "child_count": len(children), "reviewed_children": [k for k, v in children.items() if v.get("review")],
            "used_children": [k for k, v in children.items() if v.get("review", {}) and v["review"]["disposition"] == "used"],
            "unreviewed_children": [k for k, v in children.items() if not v.get("review")]}


def retain(directory):
    """Only acceptance-complete non-benchmark receipts, opt-in, no source/prompts/answers."""
    directory = Path(directory)
    frozen, result = read_json(directory / "frozen.json"), read_json(directory / "result.json")
    settings = frozen["settings"]
    verification = read_json(directory / "verification-config.json", {})
    eligible = (settings.get("task_kind", "user") in {"user", "development", "train"}
                and verification.get("kind") != "swebench_official" and result.get("autonomous_complete") is True)
    metadata = {"eligible": eligible, "retained": False, "statistics": observation(directory)}
    if eligible and settings.get("retain_experience", False):
        if digest(directory / "model.patch") != result["patch_sha256"] or digest(result["evidence_path"]) != result["evidence_sha256"]:
            raise ValueError("experience_requires_unchanged_accepted_artifact")
        identity = hashlib.sha256(json.dumps([frozen["task_id"], result["patch_sha256"],
                                           result["evidence_sha256"]]).encode()).hexdigest()
        record = {"schema": 1, "id": identity, "task_kind": settings.get("task_kind", "user"),
                  "patch_sha256": result["patch_sha256"], "verification_kind": result["verification_kind"],
                  "statistics": metadata["statistics"], "contains_source_or_trajectory": False,
                  "automatic_policy_promotion": False}
        path = Path(settings.get("experience_directory", str(directory.parent / "eligible-experience"))) / (identity + ".json")
        if not path.exists():
            atomic_json(path, record)
        metadata.update(retained=True, identity=identity)
    atomic_json(directory / "experience-status.json", metadata)
    return metadata
