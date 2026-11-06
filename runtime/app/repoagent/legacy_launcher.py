"""Prepare or run a new-project demo with the isolated final-harness repair.

Preparation is offline. Only --run starts Docker/model execution. The destination
must not exist; successful artifacts are published there after independent checks.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HARNESS = ROOT / "runtime"
# Build this local tag explicitly; never pull images implicitly.
DEFAULT_IMAGE = "repoagent-sandbox:local"
PLAN_REQUEST = """

Demo delivery instructions (in addition to the user's requirements):
Use todo_write early to save your intent and 3-6 bounded tasks as structured data,
including requirement IDs, dependencies and owners. Maintain the Todo as you work.
Implement real source files and meaningful automated tests; test discovery must
execute at least one test. Prefer Python standard-library unittest for this demo.
Do not install dependencies or assume network access. Include a README explaining
how to run and test the delivered project. Run the configured check command,
repair actual failures, review the requirements, and explicitly submit through
the normal submission tool protocol. Do not write a fake result or claim success.
Planning is an aid, not a reason to loop instead of implementing the task.
"""
# unittest alone exits successfully with zero tests. Make that case fail explicitly.
DEFAULT_CHECK = (
    "python -c \"import unittest; "
    "s=unittest.defaultTestLoader.discover('tests'); "
    "assert s.countTestCases()>0, 'No tests discovered'; "
    "r=unittest.TextTestRunner(verbosity=2).run(s); "
    "raise SystemExit(0 if r.wasSuccessful() else 1)\""
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def artifact_io(harness):
    path = Path(harness) / "app/scripts/swebench_io.py"
    spec = importlib.util.spec_from_file_location("demo_artifact_io", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(task, destination, packet, harness, *, image, model=None,
            rates=(.30, 1.20, .006), dollars=.10, seconds=600, check=DEFAULT_CHECK):
    """Build a run packet without reading a key, contacting Docker or calling a model."""
    destination, packet, harness = (Path(p).resolve() for p in (destination, packet, harness))
    if not task.strip() or not check.strip() or not image:
        raise ValueError("Task, check command and image must be supplied.")
    if destination.exists() or destination.is_symlink():
        raise ValueError("Destination must not exist; choose a new project directory.")
    for parent in (harness, ROOT, packet):
        if destination == parent or parent.is_relative_to(destination):
            raise ValueError("Destination must not contain the harness or run records.")
    if packet.is_relative_to(destination) or destination.is_relative_to(packet):
        raise ValueError("Run records and delivery destination must be separate.")
    if len(rates) != 3 or any(not math.isfinite(v) or v < 0 for v in rates) or max(rates[:2]) == 0:
        raise ValueError("Supply three nonnegative tariff rates, with a positive input/output rate.")
    if not math.isfinite(dollars) or dollars <= 0 or seconds <= 0:
        raise ValueError("Budget and deadline must be positive.")
    if not (harness / "app/corecoder_mini/supervisor.py").is_file():
        raise ValueError("Final harness entrypoint is missing.")
    profile_path = harness / "API_PROVIDER_CURRENT.json"
    profile = read_json(profile_path)
    if profile.get("service") != "DeepSeek Open Platform":
        raise ValueError("This launcher expects the current DeepSeek profile.")
    packet.mkdir(parents=True, exist_ok=False)
    seed = packet / "seed"
    seed.mkdir()
    config = {
        "task_id": "demo-" + uuid.uuid4().hex[:12], "task": task.strip() + PLAN_REQUEST,
        "workspace": str(seed), "provider": profile["provider"],
        "model": model or profile["default_model"], "base_url": profile["base_url"],
        "api_key_env": profile["api_key_env"], "pricing_usd_per_million": list(rates), "image": image,
        "settings": {
            "task_intent": "implement", "task_kind": "user", "retain_experience": False,
            "step_limit": 32, "request_limit": 32, "tool_limit": 64,
            "token_limit": 600000, "dollar_limit": dollars, "agent_seconds": seconds,
            "request_seconds": 180, "tool_seconds": 120, "context_tokens": 32000,
            "max_output_tokens": 8192, "provider_stream": True,
            "reasoning_effort": "high",
            "history_target_tokens": 8000, "history_output_chars": 2000, "keep_recent_turns": 4,
            "max_subagents": 2, "child_concurrency": 2, "child_requests": 3,
            "public_check_command": check,
        },
        "verification": {"kind": "local_command", "command": check, "seconds": 180},
    }
    write_json(packet / "task.json", config)
    write_json(packet / "demo.json", {
        "destination": str(destination), "harness": str(harness), "provider_profile": str(profile_path),
        "tariffs_are_configured_estimates_not_invoices": True,
        "model_calls_during_preparation": 0,
    })
    return config


def credential_environment(profile, inherited=None):
    """Read the credential only when the user starts the run; never serialize it."""
    env = dict(os.environ if inherited is None else inherited)
    name = profile["api_key_env"]
    value = env.get(name)
    def configured_path(key):
        path = Path(profile.get(key, "missing"))
        return path if path.is_absolute() else ROOT / path
    if not value and configured_path("dotenv_file").is_file():
        from dotenv import dotenv_values
        value = dotenv_values(configured_path("dotenv_file"), interpolate=False).get(name)
    if not value and configured_path("credential_file").is_file():
        value = configured_path("credential_file").read_text(encoding="utf-8-sig").strip()
    if not value or any(c.isspace() for c in value) or not value.startswith("sk-"):
        raise ValueError("DeepSeek credential missing or invalid; configure its private environment variable.")
    env[name] = value
    env["PYTHONUTF8"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHON_DOTENV_DISABLED"] = "1"  # Key was explicitly loaded; do not discover unrelated .env files.
    return env


def export_plan(packet):
    path = packet / "run/agent/working-state.json"
    state = read_json(path) if path.is_file() else {}
    plan = {"model_plan_recorded": bool(state.get("intent") and state.get("todos")),
            "intent": state.get("intent"), "tasks": state.get("todos", []),
            "revision": state.get("revision"), "is_acceptance": False}
    write_json(packet / "plan.json", plan)
    return plan


def failure_summary(packet):
    """Export safe failure categories only, never prompts or raw tool arguments."""
    run = Path(packet) / "run"
    result = read_json(run / "result.json") if (run / "result.json").is_file() else {}
    trajectory = read_json(run / "agent/trajectory.json") if (run / "agent/trajectory.json").is_file() else {}
    argument_errors = []
    for message in trajectory.get("messages", []):
        extra = message.get("extra", {})
        if extra.get("interrupt_type") != "FormatError":
            continue
        response = extra.get("response", {})
        if not isinstance(response, dict):
            continue
        for choice in response.get("choices", []):
            for call in choice.get("message", {}).get("tool_calls", []) or []:
                function = call.get("function", {})
                arguments = function.get("arguments")
                if not isinstance(arguments, str):
                    continue
                try:
                    json.loads(arguments)
                except json.JSONDecodeError as error:
                    name = function.get("name")
                    if name not in {"todo_write", "bash", "read_file", "edit_file", "write_file",
                                    "scratchpad", "memory", "trajectory_read", "subagent"}:
                        name = "unknown"
                    argument_errors.append({"tool": name, "category": error.msg,
                                            "line": error.lineno, "column": error.colno})
    summary = {"exit_status": result.get("checkpoint_exit"), "state": result.get("state"),
               "model_submitted": result.get("model_submitted", False),
               "request_usage": result.get("request_usage", {}),
               "estimated_usd": result.get("cost_accounting", {}).get("known_usage_tariff_usd"),
               "tool_argument_errors": argument_errors,
               "executed_action_records": len(list((run / "agent/actions").glob("*.json"))),
               "automatic_paid_retry": False}
    write_json(Path(packet) / "diagnostics.json", summary)
    return summary


def publish(packet):
    """Copy exactly the checked text artifact, fail closed on drift or destination conflicts."""
    packet = Path(packet)
    meta, result = read_json(packet / "demo.json"), read_json(packet / "run/result.json")
    if not (result.get("autonomous_complete") is True and result.get("model_submitted") is True
            and result.get("verification_ok") is True and result.get("artifact_correct") is True):
        raise ValueError("Task was not autonomously submitted and independently verified; no publication.")
    run = packet / "run"
    evidence = run / "verification.json"
    if sha(evidence) != result.get("evidence_sha256") or sha(run / "model.patch") != result.get("patch_sha256"):
        raise ValueError("Verification or patch changed after grading; no publication.")
    verdict = read_json(evidence)
    if not (verdict.get("ok") is True and verdict.get("resolved") is True
            and verdict.get("patch_applied") is True and verdict.get("patch_sha256") == result["patch_sha256"]):
        raise ValueError("Verification evidence does not match the delivered patch.")
    io = artifact_io(meta["harness"])
    if io.tree_hashes(run / "baseline") != read_json(run / "frozen.json")["baseline_sha256"]:
        raise ValueError("Original baseline changed.")
    expected = (run / "model.patch").read_bytes()
    if io.filesystem_patch(run / "baseline", run / "workspace") != expected:
        raise ValueError("Workspace changed after grading; no publication.")
    destination = Path(meta["destination"])
    if destination.exists() or destination.is_symlink():
        raise ValueError("Destination now exists; no files overwritten.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.parent / ("." + destination.name + ".delivery-" + uuid.uuid4().hex[:12])
    shutil.copytree(run / "workspace", stage, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"))
    if io.filesystem_patch(run / "baseline", stage) != expected:
        raise ValueError("Staged delivery does not match the verified artifact.")
    if destination.exists() or destination.is_symlink():
        raise ValueError("Destination appeared during publication; staged files retained separately.")
    stage.rename(destination)
    write_json(packet / "delivery.json", {"published": True, "destination": str(destination),
               "patch_sha256": result["patch_sha256"], "files": io.tree_hashes(destination)})
    return destination


def progress_snapshot(packet):
    """Numeric progress for the live demo; never echo source, prompts or arguments."""
    agent = packet / "run/agent"
    def safe_read(path):
        try:
            return read_json(path)
        except (OSError, ValueError):
            return {}
    actions = [safe_read(p) for p in (agent / "actions").glob("*.json")]
    requests = [safe_read(p) for p in (agent / "requests").glob("*.json")]
    state = safe_read(agent / "working-state.json")
    return {"requests": len(requests), "tools": len(actions),
            "edits": sum(a.get("tool") in {"edit_file", "write_file"} and a.get("result", {}).get("returncode") == 0 for a in actions),
            "checks": sum(bool(a.get("verification_candidate")) for a in actions),
            "todos": len(state.get("todos", [])),
            "todos_done": sum(t.get("status") == "done" for t in state.get("todos", [])),
            "reported_tokens": sum((r.get("prompt_tokens") or 0) + (r.get("completion_tokens") or 0) for r in requests)}


def wait_with_progress(process, packet):
    started, previous = time.monotonic(), None
    while True:
        snapshot = progress_snapshot(packet)
        if snapshot != previous:
            elapsed = round(time.monotonic() - started, 1)
            print(f"[{elapsed:6.1f}s] requests={snapshot['requests']} tools={snapshot['tools']} "
                  f"edits={snapshot['edits']} checks={snapshot['checks']} "
                  f"todo={snapshot['todos_done']}/{snapshot['todos']} tokens={snapshot['reported_tokens']}", flush=True)
            with (packet / "progress.jsonl").open("a", encoding="utf-8") as output:
                output.write(json.dumps({"seconds": elapsed, **snapshot}) + "\n")
            previous = snapshot
        try:
            return process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            continue


def run_packet(packet):
    meta, config = read_json(packet / "demo.json"), read_json(packet / "task.json")
    harness = Path(meta["harness"])
    profile = read_json(meta.get("provider_profile", harness / "API_PROVIDER_CURRENT.json"))
    if any(config[k] != profile[p] for k, p in [("base_url", "base_url"), ("api_key_env", "api_key_env"), ("provider", "provider")]):
        raise ValueError("Provider profile changed; prepare a new packet.")
    if (packet / "run").exists():
        raise ValueError("Run already exists; model calls are never automatically replayed.")
    env = credential_environment(profile) if config.get('provider') != 'ollama' else {
        **os.environ, 'PYTHONUTF8': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHON_DOTENV_DISABLED': '1'}
    print("Starting paid model execution with the configured hard budget.", flush=True)
    command = [sys.executable, "-B", "-m", "corecoder_mini.supervisor", "--config", str(packet / "task.json"),
               "--output", str(packet / "run")]
    process = subprocess.Popen(command, cwd=harness / "app", env=env)
    try:
        code = wait_with_progress(process, packet)
    except KeyboardInterrupt:
        if (packet / "run").is_dir():
            (packet / "run/STOP").touch()
        print("Stop requested. Preserve the run directory; do not restart or replay this task.", flush=True)
        return 130
    finally:
        plan = export_plan(packet)
        print("Model plan recorded: " + str(plan["model_plan_recorded"]) +
              "; tasks: " + str(len(plan["tasks"])))
    if code:
        diagnosis = failure_summary(packet)
        print("Exit reason: " + str(diagnosis["exit_status"]) +
              "; malformed tool arguments: " + str(len(diagnosis["tool_argument_errors"])) +
              "; action records: " + str(diagnosis["executed_action_records"]))
        print("Safe diagnostics: " + str(packet / "diagnostics.json"))
        print("Task incomplete or rejected. Artifacts retained; destination was not published.")
        return code
    destination = publish(packet)
    print("Verified files published to: " + str(destination))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", help="Natural-language task; prompted when omitted")
    parser.add_argument("--directory", type=Path, help="New project delivery path; must not exist")
    parser.add_argument("--harness", type=Path, default=DEFAULT_HARNESS)
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="An already installed Docker image; no pulls")
    parser.add_argument("--model", help="Default: current DeepSeek profile model")
    parser.add_argument("--rates", type=float, nargs=3, default=[.30, 1.20, .006],
                        metavar=("INPUT", "OUTPUT", "CACHE"), help="USD/million tokens; defaults use peak rates checked 2026-11-05; verify rates if changed")
    parser.add_argument("--usd", type=float, default=.10)
    parser.add_argument("--seconds", type=int, default=600)
    parser.add_argument("--check", default=DEFAULT_CHECK)
    parser.add_argument("--packet", type=Path, help="New preparation directory")
    parser.add_argument("--run", action="store_true", help="Start Docker and paid API execution")
    parser.add_argument("--run-packet", type=Path, help="Execute a previously prepared packet (paid API)")
    args = parser.parse_args(argv)
    try:
        if args.run_packet:
            return run_packet(args.run_packet.resolve())
        task = args.task or input("Task / 任务: ")
        destination = args.directory or Path(input("New project directory / 新项目目录: "))
        packet = (args.packet or ROOT / "runs" / uuid.uuid4().hex[:12]).resolve()
        prepare(task, destination, packet, args.harness, image=args.image, model=args.model,
                rates=args.rates, dollars=args.usd, seconds=args.seconds, check=args.check)
        print("Prepared configuration: " + str(packet / "task.json"))
        print("During execution, files appear in: " + str(packet / "run/workspace"))
        print("Model plan after execution: " + str(packet / "plan.json"))
        print("Configured tariffs use checked peak prices; actual charges depend on provider usage and billing.")
        if args.run:
            return run_packet(packet)
        print("No model or Docker calls. To execute: --run-packet \"" + str(packet) + "\"")
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI boundary must not echo private exception payloads
        # Provider and private-file exception bodies must never reach the terminal.
        if isinstance(exc, ValueError):
            print("Stopped: " + str(exc), file=sys.stderr)
        else:
            print("Stopped: " + type(exc).__name__ + "; inspect local run status.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
