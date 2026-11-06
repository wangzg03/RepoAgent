"""Freeze -> one upstream agent -> always export -> independent verification."""

import argparse
import ctypes
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path
from urllib.parse import urlparse

from scripts.swebench_io import tree_hashes

from .delivery import (
    atomic_bytes,
    bind_verdict,
    digest,
    export_artifact,
    lease,
    read_json,
)
from .engine import ScopedDocker
from .task_state import atomic_json

APP = Path(__file__).resolve().parents[1]
RELEASE = APP.parent
DEFAULTS = {"step_limit": 24, "tool_limit": 48, "request_limit": 24, "token_limit": 300000,
            "dollar_limit": .1, "agent_seconds": 3600, "request_seconds": 600, "tool_seconds": 180,
            "context_tokens": 16384, "max_output_tokens": 1024, "min_available_gib": 3,
            "max_memory_percent": 90, "pressure_seconds": 10}


def validate_extensions(settings):
    from .memory_embeddings import validate_memory_settings
    validate_memory_settings(settings)
    from .state_tools import TASK_KINDS
    if settings.get('task_intent') is not None and (not isinstance(settings['task_intent'], str) or settings['task_intent'] not in TASK_KINDS):
        raise ValueError('invalid_task_intent')
    if settings.get('task_intent_source', 'operator') not in {'operator', 'swebench_task_definition'}:
        raise ValueError('invalid_task_intent_source')
    if type(settings.setdefault("provider_stream", True)) is not bool:
        raise ValueError("provider_stream must be boolean")
    for name, default, minimum, maximum in [("max_subagents", 2, 0, 2), ("child_concurrency", 2, 1, 2),
            ("child_requests", 3, 1, 6), ("child_seconds", 120, 1, 600), ("keep_recent_turns", 4, 2, 12),
            ("main_request_reserve", 2, 1, 8), ("history_target_tokens", 8000, 512, 200000)]:
        value = settings.setdefault(name, default)
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError("Invalid extension setting: " + name)
    for name, default, minimum, maximum in [("stall_request_limit", 6, 0, 50),
            ("repair_window_requests", 3, 1, 12), ("max_repair_windows", 2, 1, 4)]:
        value = settings.setdefault(name, default)
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError("Invalid recovery setting: " + name)
    chars = settings.setdefault("history_output_chars", 2000)
    if type(chars) is not int or chars != 0 and not 512 <= chars <= 18000:
        raise ValueError("history_output_chars must be 0 or 512..18000")
    if type(settings.setdefault("budget_guidance", True)) is not bool:
        raise ValueError("budget_guidance must be boolean")
    names = settings.setdefault("harness_tools", ["scratchpad", "todo_write", "memory", "trajectory_read", "subagent"])
    if not isinstance(names, list) or len(names) != len(set(names)) or set(names) - {
            "scratchpad", "todo_write", "memory", "trajectory_read", "subagent", "read_file", "edit_file", "write_file"}:
        raise ValueError("Invalid harness tool allowlist")
    extensions = settings.setdefault("tool_extensions", [])
    if not isinstance(extensions, list) or not all(isinstance(x, str) for x in extensions):
        raise ValueError("Invalid installed extension allowlist")
    if settings.setdefault("task_kind", "user") not in {"user", "development", "train", "benchmark"}:
        raise ValueError("Invalid task kind")
    if type(settings.setdefault("retain_experience", False)) is not bool:
        raise ValueError("retain_experience must be boolean")
    if type(settings.setdefault("provider_attempts", 3)) is not int or not 1 <= settings["provider_attempts"] <= 3:
        raise ValueError("provider_attempts must be 1..3")
    if type(settings.setdefault("provider_retry_limit", 3)) is not int or not 0 <= settings["provider_retry_limit"] <= 3:
        raise ValueError("provider_retry_limit must be 0..3")
    if settings.get("reasoning_effort") not in {None, "none", "minimal", "low", "medium", "high"}:
        raise ValueError("invalid explicit reasoning_effort")
    from .provider_parameters import completion_parameters
    completion_parameters(settings)
    if settings.get('repoagent'):
        from repoagent.permissions import validate_settings
        validate_settings(settings)
    command = settings.get("public_check_command")
    if command is not None and (not isinstance(command, str) or not 1 <= len(command.strip()) <= 8000
                                or "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in command):
        raise ValueError("invalid_public_check_command")


def model_descriptor(config, settings):
    name = config.get("model", "qwen3.5:4b")
    provider = config.get("provider", "ollama")
    if provider == "openai":
        from .accounting import validate_rates
        endpoint = config.get("base_url", "")
        url = urlparse(endpoint)
        if url.username or url.password or url.query or url.fragment or not (
            url.scheme == "https" and url.hostname or url.scheme == "http" and url.hostname in {"localhost", "127.0.0.1", "::1"}
        ):
            raise ValueError("explicit_valid_provider_url_required_without_credentials")
        rates = config.get("pricing_usd_per_million")
        try:
            validate_rates(rates)
        except ValueError as exc:
            raise ValueError("explicit_valid_provider_pricing_required") from exc
        env_name = config.get("api_key_env", "OPENAI_API_KEY")
        import re
        if not isinstance(env_name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", env_name):
            raise ValueError("invalid_credential_environment_name")
        return {"provider": provider, "name": name, "base_url": endpoint, "api_key_env": env_name,
                "pricing_usd_per_million": rates, "digest": None, "weight_identity": "remote_provider_unverifiable"}
    if provider != "ollama":
        raise ValueError("unsupported_provider")
    tags = ollama("tags")["models"]
    model = next((m for m in tags if m["name"] == name), None)
    if not model:
        raise ValueError("Local model is not installed; no automatic download")
    info = ollama("show", {"model": name})
    if info.get("system") or info.get("messages"):
        raise ValueError("Use a local model with no inherited system prompt/history")
    parameters = info.get("parameters", "")
    import re
    match = re.search(r"\bnum_ctx\s+(\d+)", parameters)
    if not match or int(match[1]) != settings["context_tokens"]:
        raise ValueError("Model alias num_ctx must equal the configured context_tokens")
    return {"provider": provider, "name": name, "digest": model["digest"],
            "details": model.get("details"), "parameters": parameters}


def ollama(endpoint, body=None):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request("http://127.0.0.1:11434/api/" + endpoint, data=data,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def source_hashes():
    return {p.relative_to(APP).as_posix(): digest(p) for p in APP.rglob("*")
            if p.is_file() and not set(p.parts) & {"__pycache__", ".pytest_cache", ".ruff_cache"}}


def runtime_versions():
    from importlib.metadata import version
    return {"python": sys.version.split()[0], **{name: version(name) for name in ["litellm", "openai", "tenacity", "pydantic"]}}


def verify_source():
    for item in read_json(RELEASE / "UPSTREAM_SOURCES.json")["files"]:
        if digest(RELEASE / item["path"]) != item["sha256"]:
            raise ValueError("vendored_upstream_drift")
    for name, expected in read_json(RELEASE / "REFERENCE_SOURCES.json", {}).get("files", {}).items():
        if digest(RELEASE / name) != expected:
            raise ValueError("reference_source_drift")


def freeze(config, directory):
    verify_source()
    directory = Path(directory).resolve()
    settings = {**DEFAULTS, **config.get("settings", {})}
    if config.get("verification", {}).get("kind") == "swebench_official":
        if settings.get('task_intent') is not None and settings['task_intent'] != 'fix':
            raise ValueError('swebench_fix_conflicts_with_declared_task_intent')
        settings["task_kind"] = "benchmark"
        settings["retain_experience"] = False
        settings['task_intent'] = 'fix'
        settings['task_intent_source'] = 'swebench_task_definition'
    validate_extensions(settings)
    for key in DEFAULTS:
        if not isinstance(settings[key], (float, int)) or isinstance(settings[key], bool) or settings[key] <= 0:
            raise ValueError("All runtime limits must be positive: " + key)
    if settings["max_output_tokens"] + 1024 > settings["context_tokens"]:
        raise ValueError("context too small")
    source = Path(config["workspace"]).resolve(strict=True)
    if settings.get('repoagent'):
        operator = settings.get('operator_workspace', {})
        settings['project_scope'] = str(Path(operator.get('source') or operator.get('destination') or source).resolve())
    else:
        settings["project_scope"] = str(source)
    settings.setdefault("memory_path", str(directory.parent / "project-memory" / (digest_path(source) + ".sqlite3")))
    if settings["task_kind"] == "benchmark":
        settings["memory_path"] = str(directory / "quarantined-memory.sqlite3")
    if directory == source or directory.is_relative_to(source) or source.is_relative_to(directory):
        raise ValueError("Output must be separate from the input workspace")
    model = model_descriptor(config, settings)
    image = subprocess.run(["docker", "image", "inspect", config["image"], "--format", "{{.Id}}"],
                           capture_output=True, text=True, timeout=30, check=True).stdout.strip()
    directory.mkdir(parents=True, exist_ok=False)
    try:
        shutil.copytree(source, directory / "baseline", ignore=shutil.ignore_patterns(
            ".git", ".venv", "node_modules", "__pycache__", ".pytest_cache", ".env*", "key.txt", "credentials.json", "id_rsa", "id_ed25519"))
        shutil.copytree(directory / "baseline", directory / "workspace")
        atomic_json(directory / "frozen.json", {"schema": 1, "task_id": config.get("task_id", uuid.uuid4().hex),
                    "job_id": uuid.uuid4().hex, "public_task": config["task"], "settings": settings,
                    "image_id": image, "model": model,
                    "baseline_sha256": tree_hashes(directory / "baseline"), "source_sha256": source_hashes(),
                    "runtime_versions": runtime_versions(),
                    "upstream_commit": read_json(RELEASE / "UPSTREAM_SOURCES.json")["commit"],
                    "learning_enabled": False, "paid_fallback": False, "created": time.time()})
        verification = dict(config.get("verification", {}))
        if verification.get('kind') == 'humaneval_official':
            inspected = subprocess.run(['docker', 'image', 'inspect', verification['scorer_image'], '--format', '{{.Id}}'],
                                       capture_output=True, text=True, timeout=30, check=True)
            verification['scorer_image'] = inspected.stdout.strip()
        if verification.get("kind") == "swebench_official":
            from .swebench import prepare
            verification = prepare(directory, verification, config["task_id"])
        atomic_json(directory / "verification-config.json", verification)
        frozen = read_json(directory / "frozen.json")
        frozen["verification_config_sha256"] = digest(directory / "verification-config.json")
        atomic_json(directory / "frozen.json", frozen)
        shutil.make_archive(str(directory / "source"), "zip", APP)
        atomic_json(directory / "result.json", {"state": "frozen", "autonomous_complete": False,
                                               "official_correct": None})
    except BaseException:
        atomic_json(directory / "prepare-error.json", {"state": "prepare_failed"})
        raise
    return directory


def digest_path(path):
    import hashlib
    return hashlib.sha256(str(path).casefold().encode()).hexdigest()[:24]


def memory_status():
    # Keep resource checks available without a new Python dependency.
    if os.name == "nt":
        class Memory(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                        *[(n, ctypes.c_ulonglong) for n in ("total", "available", "page_total", "page_available",
                                                           "virtual_total", "virtual_available", "extended")]]
        value = Memory()
        value.length = ctypes.sizeof(value)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
            raise OSError("Cannot read host memory")
        return value.available / 2**30, value.load
    values = {line.split(":")[0]: int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()}
    available, total = values["MemAvailable"], values["MemTotal"]
    return available / 2**20, (1 - available / total) * 100


def cleanup_owned(job):
    ids = subprocess.run(["docker", "ps", "-aq", "--filter", "label=corecoder.eval.job=" + job],
                         capture_output=True, text=True, timeout=15, check=True).stdout.split()
    if ids:
        subprocess.run(["docker", "rm", "-f", *ids], capture_output=True, timeout=20, check=True)


def stop_child(process, directory):
    atomic_bytes(Path(directory) / "STOP", b"cancel\n")
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, timeout=10, check=False)
        else:
            import signal
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def run_inference(directory):
    directory = Path(directory)
    frozen = read_json(directory / "frozen.json")
    if source_hashes() != frozen["source_sha256"]:
        raise ValueError("source drift; use a new output directory")
    if frozen.get("runtime_versions") and runtime_versions() != frozen["runtime_versions"]:
        raise ValueError("runtime dependency drift; use a new output directory")
    if frozen["model"].get("provider", "ollama") == "ollama":
        installed = ollama("tags")["models"]
        if not any(m["name"] == frozen["model"]["name"] and m["digest"] == frozen["model"]["digest"] for m in installed):
            raise ValueError("model digest drift")
    settings, started = frozen["settings"], time.monotonic()
    available, load = memory_status()
    if available < settings["min_available_gib"] or load > settings["max_memory_percent"]:
        return {"reason": "resource_preflight", "inference_started": False}
    atomic_json(directory / "result.json", {"state": "running", "autonomous_complete": False,
                                           "official_correct": None})
    kwargs = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    with (directory / "worker.log").open("wb") as log:
        process = subprocess.Popen([sys.executable, "-m", "corecoder_mini.worker", str(directory)],
                                   cwd=APP, stdout=log, stderr=log, **kwargs)
        atomic_json(directory / "owned-process.json", {"pid": process.pid, "job_id": frozen["job_id"]})
        reason, pressure = "worker_exit", None
        try:
            while process.poll() is None:
                available, load = memory_status()
                unsafe = available < settings["min_available_gib"] or load > settings["max_memory_percent"]
                pressure = (pressure or time.monotonic()) if unsafe else None
                atomic_json(directory / "heartbeat.json", {"seconds": time.monotonic() - started,
                            "available_gib": available, "memory_percent": load, "pid": process.pid})
                if pressure and time.monotonic() - pressure >= settings["pressure_seconds"]:
                    reason = "sustained_memory_pressure"
                    break
                if time.monotonic() - started >= settings["agent_seconds"] + 5:
                    reason = "task_deadline"
                    break
                if (directory / "STOP").exists():
                    reason = "user_cancelled"
                    break
                time.sleep(.5)
        except KeyboardInterrupt:
            reason = "user_cancelled"
        finally:
            if process.poll() is None:
                stop_child(process, directory)
            cleanup_owned(frozen["job_id"])
    return {"reason": reason, "exit_code": process.returncode, "seconds": time.monotonic() - started,
            "inference_started": True}


def verify_artifact(directory):
    directory = Path(directory)
    frozen = read_json(directory / "frozen.json")
    if digest(directory / "verification-config.json") != frozen["verification_config_sha256"]:
        raise ValueError("verification configuration drift")
    config = read_json(directory / "verification-config.json", {})
    result = read_json(directory / "result.json")
    if config.get('kind') == 'humaneval_official':
        from repoagent.benchmarks import verify_humaneval
        return verify_humaneval(directory, config)
    if config.get("kind") == "swebench_official":
        from .swebench import verify
        return verify(directory, config)
    if config.get('kind') == 'repoagent_contract':
        from repoagent.contracts import Intent, check_command
        contract = read_json(directory / 'agent/task-contract.json')
        if not contract:
            raise ValueError('model_did_not_produce_task_contract')
        intent = Intent.model_validate(contract['intent'])
        if intent.kind in {'explain', 'locate', 'review'}:
            raise ValueError('read_only_findings_require_review; automated_code_correctness_not_scored')
        config = {'kind': 'local_command', 'command': check_command(intent), 'seconds': 180}
        atomic_json(directory / 'effective-verification.json', config)
    if config.get("kind") == "local_command":
        frozen = read_json(directory / "frozen.json")
        # Reconstruct the exact exported artifact, not a child's stale test claim.
        target = directory / ("verification-workspace-" + uuid.uuid4().hex)
        shutil.copytree(directory / "baseline", target)
        patch = (directory / "model.patch").read_bytes().decode("utf-8")
        sandbox = ScopedDocker(target, frozen["image_id"], frozen["job_id"])
        applied = sandbox.run("git apply -", target, 30, input_text=patch) if patch else None
        ok_apply = applied is None or applied.returncode == 0
        observed = sandbox.run(config["command"], target, config.get("seconds", 180)) if ok_apply else applied
        evidence = directory / "verification.json"
        verdict = {"ok": True, "resolved": ok_apply and observed.returncode == 0, "patch_applied": ok_apply,
                   "returncode": observed.returncode, "patch_sha256": result["patch_sha256"],
                   "output": observed.stdout + observed.stderr, "command": config["command"]}
        atomic_json(evidence, verdict)
        return bind_verdict(directory, verdict, evidence, kind="local_command")
    return result


def finalize(directory, terminal):
    # First persist a result even if patch generation or the scorer subsequently fails.
    directory = Path(directory)
    atomic_json(directory / "terminal.json", terminal)
    try:
        result = export_artifact(directory, terminal)
    except Exception as exc:  # noqa: BLE001 - supervisor must persist any export failure
        result = {"state": "export_failed", "error_type": type(exc).__name__, "terminal": terminal,
                  "autonomous_complete": False, "official_correct": None}
        atomic_json(directory / "result.json", result)
        return result
    frozen = read_json(directory / "frozen.json")
    if terminal.get("inference_started") and frozen["model"].get("provider", "ollama") == "ollama":
        # Unload only the alias this run selected; never stop the Ollama server.
        try:
            ollama("generate", {"model": frozen["model"]["name"], "keep_alive": 0})
        except Exception as exc:  # noqa: BLE001 - unload failure cannot erase delivery
            atomic_json(directory / "unload-warning.json", {"error_type": type(exc).__name__})
    try:
        result = verify_artifact(directory)
        from .experience import retain
        try:
            retain(directory)
        except Exception as exc:  # noqa: BLE001 - optional experience storage must never rewrite grading
            atomic_json(directory / "experience-error.json", {"error_type": type(exc).__name__})
        return result
    except Exception as exc:  # noqa: BLE001 - infrastructure errors are unscored, not passes
        result.update(state="unscored", verification_error_type=type(exc).__name__,
                      official_correct=None, autonomous_complete=False)
        atomic_json(directory / "result.json", result)
        return result


def run(directory, *, recover=False):
    directory = Path(directory).resolve()
    with lease(directory):
        if recover:
            # Only finalize after orphaned model worker has stopped. Never replay writes.
            owned = read_json(directory / "owned-process.json", {})
            if owned.get("pid"):
                try:
                    import psutil
                    process = psutil.Process(owned["pid"])
                    if "corecoder_mini.worker" in process.cmdline() and str(directory) in process.cmdline():
                        raise RuntimeError("Owned inference worker still alive; stop it before finalization")
                except ImportError as exc:
                    raise RuntimeError("Recovery process inspection requires psutil") from exc
                except psutil.NoSuchProcess:
                    pass
            previous = read_json(directory / "result.json", {})
            if previous.get("state") in {"completed", "verified", "rejected", "artifact_passed_incomplete"}:
                if digest(directory / "model.patch") != previous["patch_sha256"]:
                    raise ValueError("completed artifact drift")
                if digest(previous["evidence_path"]) != previous["evidence_sha256"]:
                    raise ValueError("completed evidence drift")
                return previous
            cleanup_owned(read_json(directory / "frozen.json")["job_id"])
            terminal = read_json(directory / "terminal.json", {"reason": "supervisor_recovery", "inference_started": True})
        else:
            if read_json(directory / "result.json", {}).get("state") != "frozen":
                raise ValueError("Inference cannot be replayed; --recover only exports/scores")
            try:
                terminal = run_inference(directory)
            except Exception as exc:  # noqa: BLE001 - retain artifacts for any worker failure
                terminal = {"reason": "supervisor_error", "error_type": type(exc).__name__, "inference_started": True}
        return finalize(directory, terminal)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--recover", action="store_true", help="Export/verify only; never resume model calls")
    args = parser.parse_args()
    if not args.recover:
        if not args.config:
            parser.error("--config required for a new run")
        freeze(read_json(args.config), args.output)
    result = run(args.output, recover=args.recover)
    print(json.dumps({k: result.get(k) for k in ("state", "model_submitted", "official_correct", "autonomous_complete")},
                     ensure_ascii=False))
    return 0 if result.get("autonomous_complete") else 1


if __name__ == "__main__":
    raise SystemExit(main())
