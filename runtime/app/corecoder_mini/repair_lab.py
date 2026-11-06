"""ReCreate outer-loop candidate staging, separate from an active task's repair loop.

No model requests, promotion, source edits or benchmark learning. The original
ScaffoldManager handles isolated version history; UTF-8 is explicit on Windows.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from . import upstream as _upstream  # noqa: F401
from .delivery import digest, read_json
from .task_state import atomic_json

EDITABLE = {"system_template", "instance_template", "action_observation_template", "format_error_template"}


def stage(run_directory, destination, proposal):
    run_directory, destination = Path(run_directory).resolve(), Path(destination).resolve()
    if destination.exists() or destination.is_relative_to(run_directory):
        raise ValueError("candidate_must_be_new_and_outside_episode")
    # Reuse the existing eligibility and integrity checks. Held-out/SWE traces
    # cannot be exported into candidates, even if they happened to pass.
    frozen, result = read_json(run_directory / "frozen.json"), read_json(run_directory / "result.json")
    if frozen["settings"].get("task_kind", "user") not in {"user", "development", "train"}:
        raise ValueError("benchmark_episode_is_not_outer_loop_training")
    verification = read_json(run_directory / "verification-config.json", {})
    if (verification.get("kind") == "swebench_official" or not result.get("autonomous_complete")
            or result.get("artifact_correct") is not True):
        raise ValueError("outer_loop_requires_accepted_development_episode")
    if (digest(run_directory / "model.patch") != result["patch_sha256"]
            or digest(result["evidence_path"]) != result["evidence_sha256"]):
        raise ValueError("accepted_episode_drift")
    changes, actions = proposal.get("changes", {}), proposal.get("evidence_actions", [])
    if (not changes or set(changes) - EDITABLE or not proposal.get("reason", "").strip()
            or len(proposal["reason"]) > 2000 or not actions or len(actions) > 8
            or any(not isinstance(v, str) or not 1 <= len(v) <= 16000 for v in changes.values())):
        raise ValueError("bounded_evidence_backed_prompt_proposal_required")
    evidence = []
    for action in actions:
        if type(action) is not int or action < 1:
            raise ValueError("actual_action_ids_required")
        path = run_directory / "agent/actions" / f"{action:04d}.json"
        receipt = read_json(path, {})
        if receipt.get("status") != "executed":
            raise ValueError("actual_executed_evidence_required")
        evidence.append({"action": action, "sha256": digest(path), "returncode": receipt["result"]["returncode"]})
    config = read_json(run_directory / "agent/trajectory.json", {}).get("info", {}).get("config", {}).get("agent", {})
    if not config.get("system_template") or not config.get("instance_template"):
        raise ValueError("frozen_scaffold_configuration_required")
    destination.mkdir(parents=True)
    atomic_json(destination / "proposal.json", {"changes": changes, "reason": proposal["reason"], "evidence": evidence})
    atomic_json(destination / "base.json", {"system_template": config["system_template"],
        "instance_template": config["instance_template"], "step_limit": frozen["settings"]["step_limit"],
        "cost_limit": frozen["settings"]["dollar_limit"]})
    # Original upstream I/O uses the locale default: force UTF-8 for this owned
    # local file-writing subprocess, never alter process-global encoding settings.
    subprocess.run([sys.executable, "-X", "utf8", "-B", "-m", "corecoder_mini.repair_lab",
                    "--write-scaffold", str(destination)], cwd=Path(__file__).resolve().parents[1],
                   check=True, capture_output=True, timeout=15)
    record = {"schema": 1, "state": "candidate_pending_equal_budget_validation", "active": False,
              "parent_run": str(run_directory), "parent_source_sha256": frozen.get("source_sha256"),
              "parent_patch_sha256": result["patch_sha256"], "evidence": evidence,
              "proposal_sha256": digest(destination / "proposal.json"),
              "scaffold_sha256": digest(destination / "scaffold/current_scaffold.yaml"),
              "automatic_promotion": False, "provider_calls": 0}
    atomic_json(destination / "candidate.json", record)
    return record


def _write(directory):
    import importlib
    module = importlib.import_module("recreate_scaffold")
    directory = Path(directory)
    manager = module.ScaffoldManager(directory / "scaffold")
    config = module.ScaffoldConfig(**read_json(directory / "base.json"))
    manager.save(config)
    for name, value in read_json(directory / "proposal.json")["changes"].items():
        setattr(config, name, value)
    manager.save(config)  # Original version/history logic, no modification to the module.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path)
    parser.add_argument("--proposal", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--write-scaffold", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.write_scaffold:
        _write(args.write_scaffold)
    elif args.episode and args.proposal and args.output:
        stage(args.episode, args.output, read_json(args.proposal))
    else:
        parser.error("episode, proposal and output required")


if __name__ == "__main__":
    main()
