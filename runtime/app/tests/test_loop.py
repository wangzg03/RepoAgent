import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from corecoder_mini.delivery import (
    bind_verdict,
    digest,
    export_artifact,
    lease,
    read_json,
)
from corecoder_mini.engine import (
    SUBMIT,
    DefaultAgent,
    DurableAgent,
    ProviderModel,
    ShellEnvironment,
    build_agent,
)
from corecoder_mini.llm import LLMResponse, ToolCall
from corecoder_mini.runtime import Cancelled, local
from corecoder_mini.supervisor import (
    DEFAULTS,
    finalize,
    run,
    run_inference,
    source_hashes,
    verify_artifact,
    verify_source,
)
from corecoder_mini.task_state import atomic_json

IMAGE = "repoagent-sandbox:local"
PREFIX = ""


@pytest.fixture(autouse=True)
def fresh_runtime():
    vars(local).clear()
    yield
    vars(local).clear()


def create_case(tmp_path):
    for name in ["baseline", "workspace"]:
        (tmp_path / name).mkdir()
        (tmp_path / name / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    from scripts.swebench_io import tree_hashes
    verification = {"kind": "local_command", "command": PREFIX +
                    "python -c 'from calc import add; assert add(2,3)==5; assert add(-1,1)==0'"}
    atomic_json(tmp_path / "verification-config.json", verification)
    atomic_json(tmp_path / "frozen.json", {"task_id": "generic-add", "image_id": IMAGE, "job_id": "testmini",
                "settings": DEFAULTS, "model": {"name": "not-used"},
                "baseline_sha256": tree_hashes(tmp_path / "baseline"),
                "verification_config_sha256": digest(tmp_path / "verification-config.json")})


class FixtureProvider:
    """Scripted outputs validate wiring, not model capability or benchmark accuracy."""
    model = "fixture"
    estimated_cost = 0
    total_prompt_tokens = total_completion_tokens = 0

    def __init__(self, commands):
        self.commands = iter(commands)
        self.wire = []

    def chat(self, messages, tools):
        self.wire.append(messages)
        assert all("extra" not in m for m in messages)
        if len(self.wire) > 1:
            assert messages[-1]["role"] == "tool"
            assert messages[-1]["tool_call_id"] == messages[-2]["tool_calls"][0]["id"]
            assert messages[-2]["reasoning_content"] == "opaque-provider-field"
        return LLMResponse(tool_calls=[ToolCall(str(len(self.wire)), "bash", {"command": next(self.commands)})],
                           reasoning_content="opaque-provider-field")


def test_actual_upstream_methods_and_integrity():
    verify_source()
    for name in ["run", "step", "query", "execute_actions"]:
        assert getattr(DurableAgent, name) is getattr(DefaultAgent, name)
    assert "vendor/minisweagent/agents/default.py" in source_hashes()


@pytest.mark.docker
def test_real_docker_failure_edit_verify_submit_and_export(tmp_path):
    create_case(tmp_path)
    check = "python -c 'from calc import add; assert add(2,3)==5'"
    edit = "python -c \"from pathlib import Path; p=Path('calc.py'); p.write_text(p.read_text().replace('a - b','a + b'))\""
    provider = FixtureProvider([check, edit, check, SUBMIT])
    agent = build_agent(provider, tmp_path / "workspace", tmp_path / "agent", IMAGE,
                        {**DEFAULTS, "shell_prefix": PREFIX, "harness_tools": []}, "mini-test-loop")
    assert agent.run("Fix add and verify its behavior.")["exit_status"] == "Submitted"
    observations = [m["extra"]["returncode"] for m in agent.messages if m["role"] == "tool"]
    assert observations == [1, 0, 0]
    exported = export_artifact(tmp_path, {"reason": "worker_exit"})
    assert exported["model_submitted"] and not exported["autonomous_complete"]
    assert exported["official_correct"] is None
    verified = verify_artifact(tmp_path)
    assert verified["autonomous_complete"] and verified["artifact_correct"]
    assert verified["official_correct"] is None
    # A local command pass is deliberately not reported as an official SWE score.
    assert verified["verification_kind"] == "local_command"


def test_supervisor_exports_without_child_result(tmp_path):
    create_case(tmp_path)
    (tmp_path / "workspace/calc.py").write_text("def add(a,b):\n    return a+b\n")
    atomic_json(tmp_path / "verification-config.json", {})
    frozen = read_json(tmp_path / "frozen.json")
    frozen["verification_config_sha256"] = digest(tmp_path / "verification-config.json")
    atomic_json(tmp_path / "frozen.json", frozen)
    result = finalize(tmp_path, {"reason": "killed", "inference_started": False})
    assert result["state"] == "exported" and result["patch_bytes"] > 0
    assert not result["autonomous_complete"] and result["official_correct"] is None
    assert not (tmp_path / "worker-result.json").exists()


def test_matching_official_score_is_required_even_after_submit(tmp_path):
    create_case(tmp_path)
    atomic_json(tmp_path / "agent/submission.json", {"model_submitted": True})
    result = export_artifact(tmp_path, {"reason": "worker_exit"})
    evidence = tmp_path / "evidence.json"
    atomic_json(evidence, {"ok": True})
    verdict = {"ok": True, "patch_applied": True, "resolved": False, "patch_sha256": result["patch_sha256"]}
    failed = bind_verdict(tmp_path, verdict, evidence, kind="swebench_official")
    assert failed["official_correct"] is False and not failed["autonomous_complete"]
    with pytest.raises(ValueError, match="scored_patch_mismatch"):
        bind_verdict(tmp_path, {**verdict, "resolved": True, "patch_sha256": "other"}, evidence,
                     kind="swebench_official")


def test_official_pass_without_model_submission_stays_incomplete(tmp_path):
    create_case(tmp_path)
    result = export_artifact(tmp_path, {"reason": "killed"})
    evidence = tmp_path / "evidence.json"
    atomic_json(evidence, {"fixture": True})
    result = bind_verdict(tmp_path, {"ok": True, "resolved": True, "patch_applied": True,
                          "patch_sha256": result["patch_sha256"]}, evidence, kind="swebench_official")
    assert result["official_correct"] is True and not result["autonomous_complete"]


def test_source_baseline_and_config_tampering_rejected(tmp_path):
    create_case(tmp_path)
    (tmp_path / "baseline/calc.py").write_text("changed")
    with pytest.raises(ValueError, match="immutable_baseline_drift"):
        export_artifact(tmp_path, {})
    atomic_json(tmp_path / "verification-config.json", {"kind": "local_command", "command": "true"})
    with pytest.raises(ValueError, match="verification configuration drift"):
        verify_artifact(tmp_path)


def test_context_overflow_stops_before_transport(tmp_path):
    provider = FixtureProvider([SUBMIT])
    model = ProviderModel(provider, tmp_path, context_tokens=2048, max_output_tokens=1024)
    with pytest.raises(RuntimeError, match="context_capacity"):
        model.query([{"role": "user", "content": "long " * 5000}])
    assert not provider.wire


def test_tool_timeout_is_observable_and_cancel_is_not_replayed(tmp_path):
    class Sandbox:
        root = tmp_path

        def run(self, command, cwd, timeout):
            if command == "slow":
                raise subprocess.TimeoutExpired(command, timeout)
            raise Cancelled("stop")

    env = ShellEnvironment(Sandbox(), tmp_path)
    assert env.execute({"command": "slow"})["returncode"] == 124
    with pytest.raises(Cancelled):
        env.execute({"command": "write"})
    receipt = read_json(tmp_path / "actions/0002.json")
    assert receipt["may_have_side_effects"] and receipt["status"] == "interrupted_or_failed"


def test_lease_is_exclusive_and_released(tmp_path):
    with lease(tmp_path), pytest.raises((OSError, BlockingIOError)), lease(tmp_path):
        pass
    with lease(tmp_path):
        pass


def test_non_standalone_submission_is_a_normal_shell_command(tmp_path):
    calls = []
    sandbox = SimpleNamespace(root=tmp_path, run=lambda *a: (calls.append(a) or subprocess.CompletedProcess(a, 0, SUBMIT, "")))
    env = ShellEnvironment(sandbox, tmp_path)
    env.execute({"command": SUBMIT + " && echo more"})
    assert calls and not (tmp_path / "submission.json").exists()


def test_real_child_crash_still_exports_patch_without_child_summary(tmp_path, monkeypatch):
    create_case(tmp_path)
    from corecoder_mini import supervisor
    frozen = read_json(tmp_path / "frozen.json")
    frozen.update(source_sha256=source_hashes(), public_task="generic", model={"name": "fixture", "digest": "frozen"})
    atomic_json(tmp_path / "frozen.json", frozen)
    atomic_json(tmp_path / "result.json", {"state": "frozen"})
    verification = {}
    atomic_json(tmp_path / "verification-config.json", verification)
    frozen["verification_config_sha256"] = digest(tmp_path / "verification-config.json")
    atomic_json(tmp_path / "frozen.json", frozen)
    real_popen = subprocess.Popen
    target = str(tmp_path / "workspace/calc.py")

    def crash_worker(argv, **kwargs):
        code = f"from pathlib import Path; import os; Path({target!r}).write_text('def add(a,b):\\n    return a+b\\n'); os._exit(17)"
        return real_popen([sys.executable, "-c", code], **kwargs)

    monkeypatch.setattr(supervisor.subprocess, "Popen", crash_worker)
    monkeypatch.setattr(supervisor, "ollama", lambda *a: {"models": [{"name": "fixture", "digest": "frozen"}]})
    monkeypatch.setattr(supervisor, "memory_status", lambda: (10, 60))
    monkeypatch.setattr(supervisor, "cleanup_owned", lambda job: None)
    result = run(tmp_path)
    assert result["terminal"]["exit_code"] == 17
    assert result["state"] == "exported" and result["patch_bytes"] > 0
    assert not result["model_submitted"] and not result["autonomous_complete"]
    assert not (tmp_path / "worker-result.json").exists()


@pytest.mark.host_process
def test_stop_request_stops_only_owned_child(tmp_path, monkeypatch):
    create_case(tmp_path)
    from corecoder_mini import supervisor
    frozen = read_json(tmp_path / "frozen.json")
    frozen.update(source_sha256=source_hashes(), model={"name": "fixture", "digest": "frozen"})
    atomic_json(tmp_path / "frozen.json", frozen)
    real_popen = subprocess.Popen

    def sleeper(argv, **kw):
        if argv[1:3] == ["-m", "corecoder_mini.worker"]:
            return real_popen([sys.executable, "-c", "import time; time.sleep(120)"], **kw)
        return real_popen(argv, **kw)

    monkeypatch.setattr(supervisor.subprocess, "Popen", sleeper)
    monkeypatch.setattr(supervisor, "ollama", lambda *a: {"models": [{"name": "fixture", "digest": "frozen"}]})
    monkeypatch.setattr(supervisor, "memory_status", lambda: (10, 60))
    cleaned = []
    monkeypatch.setattr(supervisor, "cleanup_owned", cleaned.append)
    timer = threading.Timer(.5, lambda: (tmp_path / "STOP").write_text("stop"))
    timer.start()
    started = time.monotonic()
    result = run_inference(tmp_path)
    timer.join()
    assert result["reason"] == "user_cancelled"
    assert time.monotonic() - started < 15
    assert cleaned == ["testmini"]


def test_inference_cannot_silently_restart(tmp_path):
    create_case(tmp_path)
    atomic_json(tmp_path / "result.json", {"state": "exported"})
    with pytest.raises(ValueError, match="Inference cannot be replayed"):
        run(tmp_path)


def test_incomplete_stream_does_not_execute_prefix(tmp_path):
    from types import SimpleNamespace

    from corecoder_mini.llm import LLM
    chunk = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None, reasoning_content=None,
            tool_calls=[SimpleNamespace(index=0, id="partial", function=SimpleNamespace(name="bash", arguments='{"command":"touch'))]),
            finish_reason=None)], usage=None)
    llm = LLM("fixture", "local", pricing=(0, 0))

    def broken():
        yield chunk
        raise ConnectionError("fixture disconnect")

    with pytest.raises(ConnectionError):
        llm._drain(broken(), None, None)
    assert not (tmp_path / "actions").exists()
