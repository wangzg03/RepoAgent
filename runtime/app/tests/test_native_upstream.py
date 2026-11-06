"""Exercise the real upstream transport against localhost, never a paid provider."""

import json
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from minisweagent.environments.docker import DockerEnvironment
from minisweagent.models.litellm_model import LitellmModel
from test_loop import IMAGE, PREFIX, create_case

from corecoder_mini.budget import Budget, BudgetExceeded
from corecoder_mini.delivery import export_artifact, read_json
from corecoder_mini.engine import SUBMIT, build_agent
from corecoder_mini.native_docker import SessionDocker
from corecoder_mini.native_model import NativeModel, NativeProvider, ProviderFailure
from corecoder_mini.runtime import Cancelled, local
from corecoder_mini.supervisor import DEFAULTS, freeze, run, verify_artifact
from corecoder_mini.upstream import FormatError


@pytest.fixture(autouse=True)
def clear_state():
    vars(local).clear()
    yield
    vars(local).clear()


def response(calls, *, finish_reason="tool_calls"):
    return {"id": "fixture-response", "object": "chat.completion", "created": 1, "model": "fixture",
            "choices": [{"index": 0, "finish_reason": finish_reason,
                         "message": {"role": "assistant", "content": "Local fixture response",
                                     "reasoning_content": "opaque-native-state",
                                     "tool_calls": [{"id": f"call-{i}", "type": "function",
                                                     "function": {"name": name, "arguments": json.dumps(args)}}
                                                    for i, (name, args) in enumerate(calls)]}}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70}}


@contextmanager
def local_provider(callback):
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(data)
            reply = callback(data, len(requests))
            status, result = reply[:2]
            headers = reply[2] if len(reply) > 2 else {}
            streaming = status == 200 and data.get('stream') is True
            if streaming:
                # The same scripted response over real SSE. Specialized tests
                # separately fragment calls, omit terminals/usage and drop sockets.
                choice = result['choices'][0]
                delta = dict(choice['message'])
                if delta.get('tool_calls'):
                    delta['tool_calls'] = [{**call, 'index': i} for i, call in enumerate(delta['tool_calls'])]
                first = {k: result[k] for k in ['id', 'model', 'created']}
                first.update(object='chat.completion.chunk', choices=[{
                    'index': 0, 'delta': delta, 'finish_reason': choice['finish_reason']}])
                usage = {**first, 'choices': [], 'usage': result['usage']}
                body = b''.join(b'data: ' + json.dumps(chunk).encode() + b'\n\n' for chunk in [first, usage]) + b'data: [DONE]\n\n'
            else:
                body = json.dumps(result).encode()
            self.send_response(status)
            self.send_header("Content-Type", "text/event-stream" if streaming else "application/json")
            self.send_header("Content-Length", str(len(body)))
            for name, value in headers.items():
                self.send_header(name, str(value))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (OSError, BrokenPipeError):
                pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def make_agent(tmp_path, monkeypatch, endpoint, *, extensions=None, budget=None, **settings):
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    monkeypatch.setenv("CORECODER_NATIVE_TEST_KEY", "private-fixture-credential")
    config = {**DEFAULTS, "shell_prefix": PREFIX, "harness_tools": [] if extensions is None else extensions, **settings}
    provider = NativeProvider("fixture", endpoint, "CORECODER_NATIVE_TEST_KEY", budget or Budget(.1, 100000, 12), (1, 2), config)
    return build_agent(provider, workspace, tmp_path / "agent", IMAGE, config)


def test_actual_upstream_http_preserves_schema_reasoning_and_tool_pairs(tmp_path, monkeypatch):
    def reply(data, number):
        assert data["model"] == "fixture"
        assert data["tools"][0]["function"]["name"] == "bash"
        if number == 2:
            previous = next(m for m in data["messages"] if m["role"] == "assistant")
            assert previous["reasoning_content"] == "opaque-native-state"
            assert data["messages"][-1]["role"] == "tool"
        return 200, response([("bash", {"command": "echo fixture"})])
    with local_provider(reply) as (url, requests):
        agent = make_agent(tmp_path, monkeypatch, url)
        assert isinstance(agent.model, LitellmModel)
        assert NativeModel.format_observation_messages is LitellmModel.format_observation_messages
        assert NativeModel._prepare_messages_for_api is LitellmModel._prepare_messages_for_api
        messages = [{"role": "system", "content": "Public fixture"}, {"role": "user", "content": "Inspect"}]
        reply = agent.model.query(messages)
        output = agent.model.format_observation_messages(reply, [{"returncode": 0, "output": "fixture", "exception_info": ""}])
        agent.model.query(messages + [reply] + output)
        assert len(requests) == 2 and agent.model.provider.budget.requests == 2
        assert agent.model.serialize()["info"]["provider_usage"]["prompt_tokens"] == 100
        assert read_json(tmp_path / "agent/requests/0001.json")["transport_started"]
        for p in (tmp_path / "agent").rglob("*.json"):
            assert "private-fixture-credential" not in p.read_text(encoding="utf-8")


@pytest.mark.parametrize("extensions", [[], ["scratchpad"]])
def test_new_openai_compatible_model_preserves_explicit_reasoning_effort(tmp_path, monkeypatch, extensions):
    def reply(data, number):
        assert data["model"] == "deepseek-v4.1-flash"
        assert data["reasoning_effort"] == "low"
        return 200, response([("bash", {"command": "echo local-fixture"})])
    with local_provider(reply) as (url, requests):
        agent = make_agent(tmp_path, monkeypatch, url, extensions=extensions,
                           reasoning_effort="low", provider_attempts=1)
        agent.model.config.model_name = "openai/deepseek-v4.1-flash"
        agent.model.provider.model = "deepseek-v4.1-flash"
        result = agent.model.query([{"role": "user", "content": "Public fixture"}])
        assert len(requests) == 1
        assert result["extra"]["actions"][0]["command"] == "echo local-fixture"
        assert read_json(tmp_path / "agent/requests/0001.json")["status"] == "returned"


def test_native_extensions_and_argument_failure_do_not_require_bash(tmp_path, monkeypatch):
    actions = [("scratchpad", {"action": "write", "key": "hypothesis", "value": "Public fixture"}),
               ("scratchpad", {"action": "invalid"})]
    with local_provider(lambda data, n: (200, response([actions[n - 1]]))) as (url, requests):
        agent = make_agent(tmp_path, monkeypatch, url, extensions=["scratchpad"])
        message = agent.model.query([{"role": "user", "content": "Inspect"}])
        assert message["extra"]["actions"][0]["tool"] == "scratchpad"
        with pytest.raises(FormatError):
            agent.model.query([{"role": "user", "content": "Inspect"}])
        assert len(requests) == 2 and agent.model.provider.budget.requests == 2


def test_budget_preflight_is_not_recorded_as_a_sent_request(tmp_path, monkeypatch):
    with local_provider(lambda *args: pytest.fail("Budget must block before HTTP")) as (url, requests):
        agent = make_agent(tmp_path, monkeypatch, url, budget=Budget(.1, 1, 1))
        with pytest.raises(BudgetExceeded):
            agent.model.query([{"role": "user", "content": "Inspect"}])
        receipt = read_json(tmp_path / "agent/requests/0001.json")
        assert not requests and not receipt["transport_started"]
        assert receipt["status"] == "blocked_before_transport" and receipt["usage_complete"]


def test_transport_error_has_no_unbounded_upstream_or_sdk_retry(tmp_path, monkeypatch):
    with local_provider(lambda *args: (500, {"error": {"message": "fixture failure", "type": "server_error"}})) as (url, requests):
        agent = make_agent(tmp_path, monkeypatch, url, provider_attempts=1)
        with pytest.raises(ProviderFailure):
            agent.model.query([{"role": "user", "content": "Inspect"}])
        assert len(requests) == 1 and agent.model.provider.budget.requests == 1
        assert not read_json(tmp_path / "agent/requests/0001.json")["usage_complete"]


def test_bounded_native_retry_cannot_persist_sensitive_server_error_text(tmp_path, monkeypatch, caplog):
    secret = "private-fixture-credential"
    with local_provider(lambda *args: (500, {"error": {"message": "fixture echoed " + secret, "type": "server_error"}})) as (url, requests):
        agent = make_agent(tmp_path, monkeypatch, url, provider_attempts=2)
        with pytest.raises(ProviderFailure):
            agent.run("Inspect the public fixture")
        assert len(requests) == 2 and agent.model.provider.budget.requests == 2
        assert secret not in caplog.text
        for p in (tmp_path / "agent").rglob("*.json"):
            assert secret not in p.read_text(encoding="utf-8")


def test_cancel_interrupts_native_provider_wait(tmp_path, monkeypatch):
    def delayed(*args):
        time.sleep(2)
        return 200, response([("bash", {"command": "echo late"})])
    with local_provider(delayed) as (url, _requests):
        agent = make_agent(tmp_path, monkeypatch, url)
        local.cancel = threading.Event()
        timer = threading.Timer(.15, local.cancel.set)
        timer.start()
        started = time.monotonic()
        with pytest.raises(Cancelled):
            agent.model.query([{"role": "user", "content": "Inspect"}])
        timer.join(timeout=3)
        assert time.monotonic() - started < 1.5
        assert not (tmp_path / "agent/actions").exists()


@pytest.mark.docker
def test_native_docker_keeps_tmp_bounds_output_timeout_and_git(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    docker = SessionDocker(workspace, IMAGE, "native-session-child", owner_job_id="native-session-owner")
    try:
        docker.run("echo retained >/tmp/corecoder-check", workspace, 10)
        identity = docker.environment.container_id
        assert "corecoder.eval.job=native-session-owner" in docker.environment.config.run_args
        assert "corecoder.eval.agent=native-session-child" in docker.environment.config.run_args
        assert DockerEnvironment.execute is type(docker.environment).execute
        assert docker.run("cat /tmp/corecoder-check", workspace, 10).stdout.strip() == "retained"
        assert docker.environment.container_id == identity
        denied = docker.run("touch /workspace/.git/forbidden", workspace, 10)
        assert denied.returncode != 0
        denied = docker.run("touch /root/forbidden", workspace, 10)
        assert denied.returncode != 0
        large = docker.run("yes bounded | head -c 100000", workspace, 10)
        assert len(large.stdout) < 19000 and "captured output bounded" in large.stdout
        expired = docker.run("sleep 10", workspace, 1)
        assert expired.returncode == 124
        assert docker.run("echo still-ready", workspace, 10).returncode == 0
    finally:
        docker.cleanup()
    assert docker.environment is None


@pytest.mark.docker
def test_native_command_cancel_removes_owned_container_and_preserves_other_session(tmp_path):
    one, two = tmp_path / "one", tmp_path / "two"
    one.mkdir()
    two.mkdir()
    owned, unrelated = SessionDocker(one, IMAGE, "native-cancel-owned"), SessionDocker(two, IMAGE, "native-cancel-other")
    try:
        owned.run("true", one, 10)
        unrelated.run("true", two, 10)
        local.cancel = threading.Event()
        timer = threading.Timer(.2, local.cancel.set)
        timer.start()
        started = time.monotonic()
        with pytest.raises(Cancelled):
            owned.run("sleep 30", one, 40)
        timer.join(timeout=3)
        assert owned.environment is None and time.monotonic() - started < 5
        vars(local).clear()
        assert unrelated.run("echo alive", two, 10).stdout.strip() == "alive"
    finally:
        owned.cleanup()
        unrelated.cleanup()


@pytest.mark.docker
def test_real_upstream_http_docker_edit_verify_submit_and_delivery(tmp_path, monkeypatch):
    create_case(tmp_path)
    check = "python -c 'from calc import add; assert add(2,3)==5; assert add(-1,1)==0'"
    actions = [("bash", {"command": check}),
               ("bash", {"command": "python -c \"from pathlib import Path; p=Path('calc.py'); p.write_text(p.read_text().replace('a - b','a + b'))\""}),
               ("bash", {"command": check}), ("bash", {"command": SUBMIT})]
    with local_provider(lambda data, n: (200, response([actions[n - 1]]))) as (url, requests):
        agent = make_agent(tmp_path, monkeypatch, url)
        assert agent.run("Fix the arithmetic and verify it")["exit_status"] == "Submitted"
        assert len(requests) == 4 and agent.env.registry.tools["bash"].sandbox.environment is None
    exported = export_artifact(tmp_path, {"reason": "fixture_done"})
    assert exported["model_submitted"] and not exported["autonomous_complete"]
    assert verify_artifact(tmp_path)["autonomous_complete"]
    assert read_json(tmp_path / "agent/actions/0001.json")["result"]["returncode"] != 0


@pytest.mark.docker
@pytest.mark.parametrize('qwen_compat', [False, True])
def test_native_child_uses_same_transport_budget_and_parent_integrates_result(tmp_path, monkeypatch, qwen_compat):
    create_case(tmp_path)
    counters = {"main": 0, "child": 0}
    parent_actions = [
        ("subagent", {"action": "spawn", "jobs": [{"id": "inspect", "task": "Inspect calc.py and report the arithmetic operator"}]}),
        ("subagent", {"action": "review", "identity": "inspect", "disposition": "used", "reason": "Inspect action 1 and repair in main workspace"}),
        ("bash", {"command": "python -c \"from pathlib import Path; p=Path('calc.py'); p.write_text(p.read_text().replace('a - b','a + b'))\""}),
        ("bash", {"command": "python -c 'from calc import add; assert add(2,3)==5; assert add(-1,1)==0'"}),
        ("bash", {"command": SUBMIT})]
    child_actions = [("bash", {"command": "cat calc.py"}),
                     ("subagent_report", {"summary": "Inspected arithmetic", "findings": [
                         {"claim": "add subtracts the second operand", "action": 1}]})]
    def reply(data, number):
        names = {t["function"]["name"] for t in data["tools"]}
        role = "child" if "subagent_report" in names else "main"
        counters[role] += 1
        if qwen_compat:
            assert data['max_completion_tokens'] == (1536 if role == 'child' else 8192)
            assert data['thinking_budget'] == (768 if role == 'child' else 2048)
            assert 'max_tokens' not in data and 'reasoning_effort' not in data
        if role == "child":
            assert "subagent" not in names
            if not qwen_compat:
                assert data["max_tokens"] == 1024
        return 200, response([(child_actions if role == "child" else parent_actions)[counters[role] - 1]])
    with local_provider(reply) as (url, requests):
        agent = make_agent(tmp_path, monkeypatch, url,
                           extensions=["scratchpad", "todo_write", "trajectory_read", "subagent"],
                           **({'max_output_tokens': 8192, 'context_tokens': 32000, 'output_limit_parameter': 'max_completion_tokens',
                               'temperature': .6, 'enable_thinking': True, 'preserve_thinking': True, 'thinking_budget': 2048} if qwen_compat else {}))
        assert agent.run("Fix add and verify its behavior")["exit_status"] == "Submitted"
        assert counters == {"main": 5, "child": 2}
        assert agent.model.provider.budget.requests == len(requests) == 7
        report = read_json(tmp_path / "agent/children.json")["inspect"]
        assert report["review"]["disposition"] == "used" and report["changed_paths"] == []
        assert report["actual_tools"] == ["bash", "subagent_report"]
    exported = export_artifact(tmp_path, {"reason": "fixture_done"})
    assert exported["request_usage"]["sent_requests"] == 7
    assert exported["request_usage"]["child_requests"] == 2
    assert verify_artifact(tmp_path)["autonomous_complete"]


@pytest.mark.docker
@pytest.mark.parametrize("provider_fails", [False, True])
def test_supervisor_native_worker_truthful_delivery_and_exit(tmp_path, monkeypatch, provider_fails):
    source = tmp_path / "source"
    source.mkdir()
    (source / "calc.py").write_text("def add(a,b):\n    return a-b\n")
    check = "python -c 'from calc import add; assert add(2,3)==5; assert add(-1,1)==0'"
    actions = [("bash", {"command": check}),
               ("bash", {"command": "python -c \"from pathlib import Path; p=Path('calc.py'); p.write_text(p.read_text().replace('a-b','a+b'))\""}),
               ("bash", {"command": check}), ("bash", {"command": SUBMIT})]
    def reply(data, number):
        return ((500, {"error": {"message": "fixture failure", "type": "server_error"}}) if provider_fails
                else (200, response([actions[number - 1]])))
    monkeypatch.setenv("CORECODER_NATIVE_TEST_KEY", "private-fixture-credential")
    with local_provider(reply) as (url, requests):
        output = tmp_path / "run"
        freeze({"task_id": "public-fixture", "task": "Fix add and verify the arithmetic", "workspace": str(source),
                "provider": "openai", "model": "fixture", "base_url": url, "api_key_env": "CORECODER_NATIVE_TEST_KEY",
                "pricing_usd_per_million": [1, 2], "image": IMAGE,
                "settings": {"shell_prefix": PREFIX, "harness_tools": [], "request_seconds": 10, "provider_attempts": 1},
                "verification": {"kind": "local_command", "command": PREFIX + check, "seconds": 30}}, output)
        result = run(output)
        assert result["autonomous_complete"] is (not provider_fails)
        assert result["terminal"]["exit_code"] == (1 if provider_fails else 0)
        assert result["state"] == ("rejected" if provider_fails else "completed")
        assert result["request_usage"]["sent_requests"] == len(requests) == (1 if provider_fails else 4)
        assert result["official_correct"] is None
        assert (source / "calc.py").read_text() == "def add(a,b):\n    return a-b\n"
