import copy
import json
import threading
from types import SimpleNamespace
from typing import ClassVar

import pytest
from test_loop import IMAGE, PREFIX, create_case

from corecoder_mini.budget import Budget, BudgetExceeded
from corecoder_mini.children import (
    BudgetGate,
    ChildManager,
    ChildReportTool,
    LinkedCancel,
)
from corecoder_mini.context_view import ContextView, TrajectoryTool, message_groups
from corecoder_mini.delivery import export_artifact, read_json
from corecoder_mini.engine import (
    SUBMIT,
    DefaultAgent,
    DurableAgent,
    ProviderModel,
    build_agent,
)
from corecoder_mini.events import Events
from corecoder_mini.experience import retain
from corecoder_mini.llm import LLMResponse, ToolCall
from corecoder_mini.runtime import local
from corecoder_mini.state_tools import (
    ProjectMemoryTool,
    ScratchpadTool,
    TodoTool,
    WorkingState,
)
from corecoder_mini.supervisor import DEFAULTS, validate_extensions, verify_artifact
from corecoder_mini.task_state import atomic_json
from corecoder_mini.tools.base import Tool


@pytest.fixture(autouse=True)
def runtime():
    vars(local).clear()
    yield
    vars(local).clear()


def test_draft_and_todo_survive_restart_and_failed_updates(tmp_path):
    state = WorkingState(tmp_path)
    scratch, todo = ScratchpadTool(state), TodoTool(state)
    scratch.execute("merge", value={"hypothesis": "Check the arithmetic operator", "next_experiment": "Compare expected sums"})
    todo.execute(tasks=[{"id": "locate", "content": "Locate implementation", "status": "done"},
                       {"id": "repair", "content": "Change and check", "status": "in_progress", "dependencies": ["locate"]}])
    old = copy.deepcopy(state.data)
    with pytest.raises(ValueError, match="stale"):
        scratch.execute("write", "hypothesis", "discarded", expected_revision=0)
    with pytest.raises(ValueError, match="cyclic"):
        todo.execute(tasks=[{"id": "a", "content": "A", "status": "pending", "dependencies": ["b"]},
                            {"id": "b", "content": "B", "status": "pending", "dependencies": ["a"]}])
    assert state.data == old == WorkingState(tmp_path).data
    assert "Check the arithmetic" in ScratchpadTool(WorkingState(tmp_path)).execute("read", "hypothesis")


def test_pending_dependency_cannot_be_declared_active(tmp_path):
    todo = TodoTool(WorkingState(tmp_path))
    with pytest.raises(ValueError, match="dependency_not_done"):
        todo.execute(tasks=[{"id": "a", "content": "A", "status": "pending"},
                            {"id": "b", "content": "B", "status": "in_progress", "dependencies": ["a"]}])


def test_project_memory_is_scoped_and_benchmark_is_quarantined(tmp_path):
    one = ProjectMemoryTool(tmp_path / "memory.db", "project-one")
    two = ProjectMemoryTool(tmp_path / "memory.db", "project-two")
    identity = json.loads(one.execute("save", text="Use the documented build command", key="build"))["identity"]
    assert "build command" in ProjectMemoryTool(tmp_path / "memory.db", "project-one").execute("read", identity=identity)
    assert json.loads(two.execute("search", query="build")) == []
    with pytest.raises(ValueError, match="scope"):
        two.execute("read", identity=identity)
    benchmark = ProjectMemoryTool(tmp_path / "memory.db", "project-one", enabled=False)
    assert "quarantined" in benchmark.execute("search", query="build")
    assert "quarantined" in benchmark.execute("save", text="held-out information")
    assert len(json.loads(one.execute("search", query="build"))) == 1


def history(count=10):
    messages = [{"role": "system", "content": "Stable system"}, {"role": "user", "content": "Original task must survive"}]
    for i in range(1, count + 1):
        messages.extend([{"role": "assistant", "content": "Inspect", "reasoning_content": "opaque",
                          "tool_calls": [{"id": str(i), "type": "function", "function": {
                              "name": "bash", "arguments": json.dumps({"command": "echo diagnostic"})}}]},
                         {"role": "tool", "tool_call_id": str(i), "content": f"turn-{i}:" + "source text " * 500,
                          "extra": {"returncode": 1 if i == 2 else 0}}])
    return messages


def test_projection_keeps_full_groups_failure_task_and_drafts(tmp_path):
    original = history()
    state = WorkingState(tmp_path)
    ScratchpadTool(state).execute("write", "hypothesis", "Current hypothesis must survive")
    view = ContextView(tmp_path, state, Events(tmp_path), target_tokens=1000, keep_recent=4)
    rendered, receipt = view.project(original, capacity=13000)
    assert receipt["compacted"] and {2, 7, 8, 9, 10} <= set(receipt["retained_turns"])
    assert original == history()  # Canonical messages are never rewritten.
    assert rendered[0]["content"] == "Stable system"
    assert "Original task must survive" in json.dumps(rendered)
    assert "Current hypothesis must survive" in json.dumps(rendered)
    _, groups = message_groups(rendered)
    assert all(g[0]["reasoning_content"] == "opaque" for g in groups)
    assert "turn-1:" in TrajectoryTool(tmp_path).execute(turn=1)


def test_projection_stops_instead_of_dropping_required_failure(tmp_path):
    with pytest.raises(RuntimeError, match="cannot fit safely"):
        ContextView(tmp_path, WorkingState(tmp_path), Events(tmp_path), keep_recent=4).project(history(), capacity=300)


@pytest.mark.parametrize("messages", [
    [{"role": "tool", "tool_call_id": "orphan", "content": "x"}],
    [{"role": "assistant", "tool_calls": [{"id": "missing"}]}],
])
def test_invalid_protocol_is_never_silently_repaired(messages):
    with pytest.raises(ValueError):
        message_groups(messages)


def test_shared_child_budget_preserves_main_requests():
    gate = BudgetGate(Budget(1, token_limit=100000, request_limit=5), main_reserve=2)
    params = {"model": "local", "messages": [], "max_tokens": 10}
    usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1)
    first = gate.reserve(params, (0, 0))
    gate.settle(first, usage, (0, 0))
    child = gate.fork(4)
    for _ in range(2):
        reservation = child.reserve(params, (0, 0))
        child.settle(reservation, usage, (0, 0))
    with pytest.raises(BudgetExceeded, match="main_reserve"):
        child.reserve(params, (0, 0))
    assert gate.requests == 3
    reservation = gate.reserve(params, (0, 0))
    gate.settle(reservation, usage, (0, 0))


def test_cancel_propagates_down_without_child_cancelling_parent():
    root = threading.Event()
    one, two = LinkedCancel(root), LinkedCancel(root)
    one.set()
    assert one.is_set() and not root.is_set() and not two.is_set()
    root.set()
    assert two.wait(.1)


class ScriptedProvider:
    """A controlled provider, explicitly not a real-model success measurement."""
    model = "fixture"
    estimated_cost = 0
    total_prompt_tokens = total_completion_tokens = 0

    def __init__(self, actions):
        self.actions = iter(actions)
        self.requests = []

    def chat(self, messages, tools):
        message_groups(messages)
        self.requests.append(copy.deepcopy(messages))
        name, args = next(self.actions)
        assert name in {t["function"]["name"] for t in tools}
        self.total_prompt_tokens += 10
        self.total_completion_tokens += 5
        return LLMResponse(content="Controlled test action", reasoning_content="opaque",
                           tool_calls=[ToolCall(str(len(self.requests)), name, args)], prompt_tokens=10, completion_tokens=5)


@pytest.mark.docker
def test_main_actually_uses_child_finding_edits_and_verifies(tmp_path):
    create_case(tmp_path)
    frozen = read_json(tmp_path / "frozen.json")
    frozen["settings"].update(task_kind="development", retain_experience=True,
                              experience_directory=str(tmp_path / "eligible-experience"))
    atomic_json(tmp_path / "frozen.json", frozen)
    parent = ScriptedProvider([
        ("todo_write", {"tasks": [{"id": "repair", "content": "Inspect, repair and verify add", "status": "in_progress"}]}),
        ("scratchpad", {"action": "merge", "value": {"hypothesis": "Check add implementation"}}),
        ("subagent", {"action": "spawn", "jobs": [{"id": "inspect", "task": "Inspect calc.py and report the arithmetic implementation"}]}),
        ("subagent", {"action": "review", "identity": "inspect", "disposition": "used", "reason": "Action 1 shows the wrong operator; inspect and repair in main workspace"}),
        ("bash", {"command": "python -c \"from pathlib import Path; p=Path('calc.py'); p.write_text(p.read_text().replace('a - b','a + b'))\""}),
        ("bash", {"command": "python -c 'from calc import add; assert add(2,3)==5; assert add(-1,1)==0'"}),
        ("todo_write", {"tasks": [{"id": "repair", "content": "Inspect, repair and verify add", "status": "done"}]}),
        ("bash", {"command": SUBMIT})])
    providers = {}

    def child_factory(identity, budget):
        child = ScriptedProvider([("bash", {"command": "cat calc.py"}),
                 ("subagent_report", {"summary": "Inspected calc.py", "findings": [{"claim": "add subtracts b instead of adding", "action": 1}]})])
        child.budget = None  # Factory cannot detach a child from the shared ledger.
        providers[identity] = child
        return child

    agent = build_agent(parent, tmp_path / "workspace", tmp_path / "agent", IMAGE,
                        {**DEFAULTS, "shell_prefix": PREFIX}, "mini-extended-test", child_provider_factory=child_factory)
    assert agent.run("Fix add.")["exit_status"] == "Submitted"
    report = read_json(tmp_path / "agent/children.json")["inspect"]
    assert report["review"]["disposition"] == "used" and report["actual_tools"] == ["bash", "subagent_report"]
    assert report["changed_paths"] == []
    assert providers["inspect"].budget.shared is agent.children.gate.shared
    assert all("subagent" not in m["content"] for m in providers["inspect"].requests[0] if m["role"] == "system")
    result = export_artifact(tmp_path, {"reason": "worker_exit"})
    assert result["harness_observation"]["used_children"] == ["inspect"]
    assert verify_artifact(tmp_path)["autonomous_complete"]
    assert ScratchpadTool(WorkingState(tmp_path / "agent")).execute("read", "hypothesis")
    assert retain(tmp_path)["retained"]
    assert retain(tmp_path)["retained"]
    receipts = list((tmp_path / "eligible-experience").glob("*.json"))
    assert len(receipts) == 1 and not read_json(receipts[0])["contains_source_or_trajectory"]


@pytest.mark.docker
def test_child_mount_is_read_only_and_cannot_recursively_delegate(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "source.py").write_text("original")
    provider = ScriptedProvider([("bash", {"command": "echo changed > source.py"}),
        ("subagent_report", {"summary": "Read-only boundary confirmed", "findings": [], "unresolved": "Write rejected"})])
    settings = {**DEFAULTS, "max_subagents": 0, "harness_tools": ["scratchpad", "todo_write", "trajectory_read"]}
    agent = build_agent(provider, workspace, tmp_path / "agent", IMAGE, settings, "mini-child-ro-test", depth=1)
    assert "subagent" not in agent.env.registry.tools
    assert agent.run("Inspect only")["exit_status"] == "ChildReported"
    receipt = read_json(tmp_path / "agent/actions/0001.json")
    assert receipt["result"]["returncode"] != 0 and (workspace / "source.py").read_text() == "original"


def test_lifetime_child_limit_and_dependency_review(tmp_path):
    state, events = WorkingState(tmp_path), Events(tmp_path)
    manager = ChildManager(ScriptedProvider([]), tmp_path, tmp_path, IMAGE, {**DEFAULTS, "max_subagents": 2}, state, events)
    manager.records = {"one": {"status": "completed", "report": {}, "review": None}}
    with pytest.raises(ValueError, match="parent_acceptance"):
        manager.dispatch([{"id": "two", "task": "Inspect", "dependencies": ["one"]}])
    manager.records["two"] = {"status": "failed"}
    with pytest.raises(ValueError, match="lifetime"):
        manager.dispatch([{"id": "three", "task": "Inspect"}])


def test_todo_done_does_not_make_artifact_accepted(tmp_path):
    create_case(tmp_path)
    TodoTool(WorkingState(tmp_path / "agent")).execute(tasks=[{"content": "All good", "status": "done"}])
    result = export_artifact(tmp_path, {"reason": "stopped"})
    assert not result["autonomous_complete"] and result["official_correct"] is None


def test_benchmark_trajectory_never_enters_experience(tmp_path):
    create_case(tmp_path)
    frozen = read_json(tmp_path / "frozen.json")
    frozen["settings"].update(task_kind="benchmark", retain_experience=True)
    atomic_json(tmp_path / "frozen.json", frozen)
    atomic_json(tmp_path / "result.json", {"autonomous_complete": True})
    result = retain(tmp_path)
    assert not result["eligible"] and not result["retained"]


def test_invalid_extension_settings_and_upstream_identity():
    with pytest.raises(ValueError, match="max_subagents"):
        validate_extensions({"max_subagents": 99})
    for name in ["run", "step", "query", "execute_actions"]:
        assert getattr(DurableAgent, name) is getattr(DefaultAgent, name)


def test_event_subscriber_and_records_survive_restart(tmp_path):
    events, seen = Events(tmp_path), []
    events.subscribe(lambda event: seen.append(event["type"]))
    assert events.emit("tool_started", tool="bash", action=1) == 1
    assert Events(tmp_path).emit("tool_finished", tool="bash", returncode=0) == 2
    assert seen == ["tool_started"]
    assert all("arguments" not in read_json(p) for p in (tmp_path / "events").glob("*.json"))


def test_draft_pagination_recovers_middle_content(tmp_path):
    scratch = ScratchpadTool(WorkingState(tmp_path))
    scratch.execute("write", "details", "a" * 6000 + "essential_middle_fact" + "b" * 6000)
    page = json.loads(scratch.execute("read", "details", offset=5900, limit=1000))
    assert "essential_middle_fact" in page["fragment"] and page["next_offset"] == 6900


def test_trajectory_pagination_recovers_archived_middle_content(tmp_path):
    messages = history(1)
    messages[-1]["content"] = "a" * 6000 + "essential_failure_detail" + "b" * 6000
    atomic_json(tmp_path / "canonical-messages.json", messages)
    page = json.loads(TrajectoryTool(tmp_path).execute(turn=1, offset=5900, limit=1000))[0]["observations"][0]
    assert "essential_failure_detail" in page["content"] and page["next_offset"] == 6900


def test_older_trajectory_index_can_be_paged_and_keeps_code_locations(tmp_path):
    messages = history(35)
    messages[2]["tool_calls"][0]["function"]["arguments"] = json.dumps({"command": "sed -n '1,20p' src/module.py"})
    atomic_json(tmp_path / "canonical-messages.json", messages)
    tool = TrajectoryTool(tmp_path)
    first = json.loads(tool.execute(action="index"))
    second = json.loads(tool.execute(action="index", offset=first["next_offset"]))
    assert first["index"][0]["actions"][0]["mentioned_paths"] == ["src/module.py"]
    assert second["index"][0]["turn"] == 31 and second["next_offset"] is None


def test_prompt_only_advertises_registered_extension_tools(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    agent = build_agent(ScriptedProvider([]), workspace, tmp_path / "agent", IMAGE,
                        {**DEFAULTS, "max_subagents": 0, "harness_tools": ["trajectory_read"]})
    assert "trajectory_read" in agent.config.instance_template
    for name in ["subagent", "scratchpad", "todo_write"]:
        assert name not in agent.config.instance_template and name not in agent.env.registry.tools


@pytest.mark.docker
def test_new_tool_is_visible_to_model_and_dispatched_without_changing_loop(tmp_path):
    class Echo(Tool):
        name = "echo_note"
        description = "Test extension"
        parameters: ClassVar[dict] = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}

        def execute(self, text):
            return text

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provider = ScriptedProvider([("echo_note", {"text": "extension result"}), ("bash", {"command": SUBMIT})])
    agent = build_agent(provider, workspace, tmp_path / "agent", IMAGE, {**DEFAULTS, "harness_tools": []})
    assert not (tmp_path / "agent/project-memory.sqlite3").exists()
    tool = agent.env.registry.register(Echo())
    assert not tool.read_only and not tool.concurrency_safe
    assert agent.run("Exercise extension")["exit_status"] == "Submitted"
    assert "extension result" in json.dumps(provider.requests[-1])


@pytest.mark.docker
def test_two_children_really_overlap_and_return_independent_context(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "source.py").write_text("public fixture")
    barrier, calls = threading.Barrier(2), []

    class ConcurrentProvider(ScriptedProvider):
        def chat(self, messages, tools):
            if not self.requests:
                barrier.wait(timeout=5)
            calls.append(threading.current_thread().name)
            return super().chat(messages, tools)

    def factory(identity, budget):
        child = ConcurrentProvider([("bash", {"command": "cat source.py"}),
            ("subagent_report", {"summary": "Inspected " + identity, "findings": [{"claim": "Read source fixture", "action": 1}]})])
        child.budget = budget
        return child

    session = tmp_path / "session"
    manager = ChildManager(ScriptedProvider([]), workspace, session, IMAGE, {**DEFAULTS, "child_concurrency": 2},
                           WorkingState(session), Events(session), provider_factory=factory)
    reports = manager.dispatch([{"id": "source", "task": "Inspect source"}, {"id": "tests", "task": "Inspect test interface"}])
    assert len(set(calls)) == 2 and all(r["status"] == "completed" for r in reports)
    assert all(r["changed_paths"] == [] for r in reports)


def test_parallel_child_admission_cannot_spend_main_reserved_requests():
    gate = BudgetGate(Budget(1, token_limit=100000, request_limit=3), main_reserve=2)
    barrier, outcomes = threading.Barrier(2), []
    params = {"messages": [], "max_tokens": 10}

    def reserve():
        child = gate.fork(2)
        barrier.wait(timeout=2)
        try:
            r = child.reserve(params, (0, 0))
            child.settle(r, SimpleNamespace(prompt_tokens=1, completion_tokens=1), (0, 0))
            outcomes.append("admitted")
        except BudgetExceeded:
            outcomes.append("blocked")

    threads = [threading.Thread(target=reserve) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert sorted(outcomes) == ["admitted", "blocked"] and gate.requests == 1


def test_child_cannot_invent_action_evidence(tmp_path):
    report = ChildReportTool(tmp_path)
    atomic_json(tmp_path / "actions/0001.json", {"status": "intent"})
    with pytest.raises(ValueError, match="actual_successful_action"):
        report.execute("Claim", [{"claim": "Fixed", "action": 1}])


def test_session_state_cannot_be_mounted_inside_source_workspace(tmp_path):
    with pytest.raises(ValueError, match="outside_sandbox"):
        build_agent(ScriptedProvider([]), tmp_path, tmp_path / "state", IMAGE, DEFAULTS)


def test_api_config_does_not_probe_provider_or_persist_credentials(monkeypatch):
    from corecoder_mini import supervisor
    from corecoder_mini.worker import create_provider
    monkeypatch.setattr(supervisor, "ollama", lambda *a: pytest.fail("API descriptor must not probe Ollama"))
    monkeypatch.setenv("CORECODER_TEST_CREDENTIAL", "fixture-only-not-a-real-key")
    model = supervisor.model_descriptor({"provider": "openai", "model": "test-model", "base_url": "https://api.example.test/v1",
                "api_key_env": "CORECODER_TEST_CREDENTIAL", "pricing_usd_per_million": [1, 2]}, DEFAULTS)
    assert "fixture-only-not-a-real-key" not in json.dumps(model)
    assert model["digest"] is None
    llm = create_provider(model, DEFAULTS, Budget(.1, 10000, 2))
    assert llm.base_url == "https://api.example.test/v1"
    assert "reasoning_effort" not in llm.extra
    assert "fixture-only-not-a-real-key" not in repr(llm)


def test_missing_api_credential_stops_before_transport(monkeypatch):
    from corecoder_mini.worker import create_provider
    monkeypatch.delenv("CORECODER_TEST_MISSING_CREDENTIAL", raising=False)
    with pytest.raises(ValueError, match="credential_environment_missing"):
        create_provider({"provider": "openai", "name": "fixture", "api_key_env": "CORECODER_TEST_MISSING_CREDENTIAL"},
                        DEFAULTS, Budget(.1, 10000, 2))


def test_child_request_cost_does_not_include_other_parallel_requests(tmp_path):
    provider = ScriptedProvider([("bash", {"command": "echo observed"})])
    provider.pricing = (1, 2)
    provider.estimated_cost = .09  # A shared ledger may already include another child's usage.
    model = ProviderModel(provider, tmp_path)
    response = model.query([{"role": "user", "content": "Inspect"}])
    assert response["extra"]["cost"] == pytest.approx(.000020)
    assert model.serialize()["info"]["provider_usage"]["shared_cost_usd_estimate"] == .09


@pytest.mark.docker
def test_optional_experience_failure_cannot_change_real_verdict(tmp_path, monkeypatch):
    from corecoder_mini import experience
    from corecoder_mini.supervisor import finalize
    create_case(tmp_path)
    (tmp_path / "workspace/calc.py").write_text("def add(a,b):\n    return a+b\n")
    atomic_json(tmp_path / "agent/submission.json", {"model_submitted": True})
    monkeypatch.setattr(experience, "retain", lambda directory: (_ for _ in ()).throw(OSError("fixture disk error")))
    result = finalize(tmp_path, {"reason": "test", "inference_started": False})
    assert result["autonomous_complete"] and result["state"] == "completed"
    assert (tmp_path / "experience-error.json").is_file()
