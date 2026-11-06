import copy
import json
from types import SimpleNamespace

import pytest
from test_harness import ScriptedProvider, history
from test_loop import create_case

from corecoder_mini.budget import Budget
from corecoder_mini.children import BudgetGate
from corecoder_mini.context import estimate_tokens
from corecoder_mini.context_view import ContextView, TrajectoryTool, message_groups
from corecoder_mini.delivery import export_artifact, read_json
from corecoder_mini.engine import BashTool, ProviderModel, ShellEnvironment
from corecoder_mini.events import Events
from corecoder_mini.llm import LLM, LLMResponse
from corecoder_mini.runtime import local
from corecoder_mini.state_tools import ScratchpadTool, TodoTool, WorkingState
from corecoder_mini.supervisor import validate_extensions
from corecoder_mini.token_policy import RuntimeGuidance, token_components
from corecoder_mini.tools.base import Tool


@pytest.fixture(autouse=True)
def runtime():
    vars(local).clear()
    yield
    vars(local).clear()


def test_reduce_history_preserves_failure_recent_turns_and_exact_recall(tmp_path):
    original = history(8)
    # Realistic unique source lines, not just repeated padding.
    original[3]["content"] = "\n".join(f"line_{i} = {i}" for i in range(900))
    original[3]["content"] += "\nFINAL_VALUE"
    original[5]["content"] = "failure prefix\n" + "failure detail\n" * 800 + "MIDDLE_COUNTEREXAMPLE\nend"
    state = WorkingState(tmp_path)
    ScratchpadTool(state).execute("write", "next_experiment", "Check the omitted source range")
    old = copy.deepcopy(original)
    rendered, receipt = ContextView(tmp_path, state, Events(tmp_path), target_tokens=100000,
                                    history_output_chars=1200).project(original, 100000)
    _, groups = message_groups(rendered)
    assert original == old == read_json(tmp_path / "canonical-messages.json")
    assert len(groups[0][1]["content"]) <= 1200
    assert groups[1][1]["content"] == old[5]["content"]
    assert groups[-2][1]["content"] == old[-3]["content"]
    assert groups[-1][1]["content"] == old[-1]["content"]
    assert "Check the omitted source range" in rendered[-1]["content"]
    assert all(g[0]["reasoning_content"] == "opaque" for g in groups)
    recovered = json.loads(TrajectoryTool(tmp_path).execute(turn=1, offset=6000, limit=1000))
    assert recovered[0]["observations"][0]["content"] == old[3]["content"][6000:7000]
    assert receipt["estimated_tokens_saved"] > 0 and receipt["output_reductions"]


def test_exact_duplicate_has_retained_reference_but_different_command_is_not_deduplicated(tmp_path):
    messages = history(7)
    for i in range(7):
        messages[3 + i * 2]["content"] = "\n".join(f"symbol_{n} = {n}" for n in range(250))
    messages[2]["tool_calls"][0]["function"]["arguments"] = json.dumps({"command": "cat different.py"})
    state = WorkingState(tmp_path)
    rendered, receipt = ContextView(tmp_path, state, Events(tmp_path), target_tokens=100000).project(messages, 100000)
    _, groups = message_groups(rendered)
    assert groups[0][1]["content"] == messages[3]["content"]  # Different command: keep evidence.
    duplicate = [c for c in receipt["output_reductions"] if c["method"] == "exact_duplicate"]
    assert duplicate and all("turn 7" in groups[c["turn"] - 1][1]["content"] for c in duplicate)
    assert groups[-1][1]["content"] == messages[-1]["content"]


def test_user_steering_keeps_its_chronological_position_after_archiving(tmp_path):
    messages = history(9)
    messages.insert(14, {"role": "user", "content": "New constraint: preserve unicode."})
    rendered, receipt = ContextView(tmp_path, WorkingState(tmp_path), Events(tmp_path), target_tokens=1600).project(messages, 15000)
    assert receipt["compacted"]
    steering = next(i for i, m in enumerate(rendered) if m.get("content") == "New constraint: preserve unicode.")
    preceding_ids = [m["tool_call_id"] for m in rendered[:steering] if m["role"] == "tool"]
    following_ids = [m["tool_call_id"] for m in rendered[steering:] if m["role"] == "tool"]
    assert "6" in preceding_ids and "7" in following_ids
    message_groups(rendered)


def test_no_recall_tool_means_no_unrecoverable_projection(tmp_path):
    messages = history(5)
    view = ContextView(tmp_path, WorkingState(tmp_path), Events(tmp_path), target_tokens=1000, allow_recall=False)
    rendered, receipt = view.project(messages, 100000)
    assert not receipt["compacted"] and not receipt["output_reductions"]
    assert [m["content"] for m in rendered[:-1]] == [m["content"] for m in messages]
    with pytest.raises(RuntimeError, match="trajectory_read unavailable"):
        view.project(messages, 100)


def test_original_action_output_is_paged_without_reexecuting_command(tmp_path):
    text = "BEGIN\n" + "\n".join(f"captured line {i}" for i in range(2400)) + "\nEND"

    class BigOutput(Tool):
        name = "bash"
        description = "Fixture command"
        parameters = BashTool.parameters
        calls = 0
        def execute(self, command):
            self.calls += 1
            return {"returncode": 1, "output": text}

    environment = ShellEnvironment(SimpleNamespace(), tmp_path)
    tool = BigOutput()
    environment.registry.tools["bash"] = tool
    reader = TrajectoryTool(tmp_path)
    environment.registry.register(reader)
    result = environment.execute({"command": "fixture", "tool_call_id": "one"})
    model = ProviderModel(ScriptedProvider([]), tmp_path)
    formatted = model.format_observation_messages({"extra": {"actions": [{"tool_call_id": "one"}]}}, [result])
    assert len(formatted[0]["content"]) < len(text)
    assert "action_id=1" in formatted[0]["content"]
    recalled = json.loads(reader.execute(action="output", action_id=1, offset=15000, limit=4000))
    assert recalled["content"] == text[15000:19000] and recalled["next_offset"] == 19000
    assert tool.calls == 1 and environment.actions == 1
    with pytest.raises(ValueError, match="positive_action_id"):
        reader.execute(action="output", action_id="../secret")
    with pytest.raises(ValueError, match="not_in_this_session"):
        reader.execute(action="output", action_id=2)


def test_budget_reminders_share_usage_but_do_not_change_limits_or_call_model():
    shared = Budget(.1, token_limit=10000, request_limit=10)
    shared.spent, shared.tokens_spent, shared.requests = .085, 8400, 8
    gate = BudgetGate(shared)
    environment = SimpleNamespace(actions=6)
    llm = SimpleNamespace(budget=gate)
    settings = {"tool_limit": 48, "agent_seconds": 3600}
    before = shared.spent, shared.tokens_spent, shared.requests, shared.limit, shared.token_limit, shared.request_limit
    advice = RuntimeGuidance(environment, llm, settings)()
    assert advice["budget"] == {"tools_remaining": 42, "requests_remaining": 2, "tokens_remaining": 1600, "usd_remaining": .015}
    assert advice["advisory_only"] and "verification" in advice["guidance"] and "incomplete" in advice["guidance"]
    assert before == (shared.spent, shared.tokens_spent, shared.requests, shared.limit, shared.token_limit, shared.request_limit)


def test_early_tasks_do_not_pay_for_unneeded_budget_reminders():
    shared = Budget(.1, token_limit=10000, request_limit=10)
    assert RuntimeGuidance(SimpleNamespace(actions=1), SimpleNamespace(budget=BudgetGate(shared)),
                           {"tool_limit": 48, "agent_seconds": 3600})() is None


def test_child_guidance_accounts_for_its_own_cap_without_spending_parent_budget():
    gate = BudgetGate(Budget(1, token_limit=100000, request_limit=24)).fork(3)
    gate.local_requests = 2
    feedback = RuntimeGuidance(SimpleNamespace(actions=1), SimpleNamespace(budget=gate),
                               {"tool_limit": 8, "agent_seconds": 120})()
    assert feedback["budget"]["child_requests_remaining"] == 1
    assert gate.requests == 0 and "targeted" in feedback["guidance"]


def test_todo_write_ack_does_not_duplicate_full_plan_in_next_observation(tmp_path):
    state = WorkingState(tmp_path)
    todo = TodoTool(state)
    tasks = [{"content": "Locate source and preserve behavior " * 12, "status": "pending"} for _ in range(8)]
    ack = json.loads(todo.execute(tasks=tasks))
    assert ack["stored_tasks"] == 8 and "todos" not in ack
    assert len(json.dumps(ack)) < 200
    assert len(json.loads(todo.execute(action="read"))["tasks"]) == 8
    assert state.view()["todos"]  # The plan remains in durable memory.


def test_request_components_and_failure_diagnosis_are_persisted(tmp_path):
    provider = ScriptedProvider([("bash", {"command": "echo check"})])
    model = ProviderModel(provider, tmp_path)
    messages = [{"role": "system", "content": "Instructions"}, {"role": "user", "content": "Public task"}]
    model.query(messages)
    receipt = read_json(tmp_path / "requests/0001.json")
    assert sum(receipt["estimated_token_components"].values()) == receipt["estimated_input_tokens"]
    assert receipt["cached_prompt_tokens"] is None

    def fail(*args, **kwargs):
        raise ConnectionError("fixture")
    provider.chat = fail
    with pytest.raises(ConnectionError):
        model.query(messages)
    failed = read_json(tmp_path / "requests/0002.json")
    assert failed["status"] == "interrupted_or_failed" and not failed["usage_complete"]
    assert sum(failed["estimated_token_components"].values()) == failed["estimated_input_tokens"]


@pytest.mark.parametrize("cached", [None, 0, 12])
def test_provider_cache_usage_is_reported_only_when_available(cached):
    llm = LLM("fixture", "unused", pricing=(0, 0))
    usage = SimpleNamespace(prompt_tokens=30, completion_tokens=5,
                            prompt_tokens_details=SimpleNamespace(cached_tokens=cached))
    response = llm._drain(iter([SimpleNamespace(choices=[], usage=usage)]), None, None)
    assert response.cached_prompt_tokens == cached
    assert response.prompt_tokens == 30 and response.completion_tokens == 5


def test_json_tool_compatibility_keeps_actual_cache_receipt():
    response = LLMResponse(content='{"name":"bash","arguments":{"command":"echo ok"}}',
                           prompt_tokens=30, completion_tokens=5, cached_prompt_tokens=12)
    adapted = LLM._text_tool_response(response, [{
        "function": {"name": "bash", "parameters": BashTool.parameters}}])
    assert adapted.tool_calls[0].name == "bash" and adapted.cached_prompt_tokens == 12


@pytest.mark.parametrize("setting", [{"history_output_chars": True}, {"history_output_chars": 200}, {"budget_guidance": 1}])
def test_invalid_token_policy_settings_fail_before_inference(setting):
    with pytest.raises(ValueError):
        validate_extensions(setting)


def test_token_components_include_reasoning_and_arguments():
    messages = history(3)
    components = token_components(messages)
    assert components["native_reasoning"] > 0 and components["tool_arguments"] > 0
    assert sum(components.values()) == estimate_tokens(messages) + len(messages) * 8


def test_delivery_reports_estimates_separately_from_actual_usage(tmp_path):
    create_case(tmp_path)
    from corecoder_mini.task_state import atomic_json
    atomic_json(tmp_path / "agent/requests/0001.json", {
        "prompt_tokens": 100, "completion_tokens": 10, "usage_complete": True,
        "cached_prompt_tokens": 20, "estimated_token_components": {"tool_observations": 90},
        "projection": {"estimated_tokens_saved": 30, "output_reductions": [{}]}})
    result = export_artifact(tmp_path, {"reason": "fixture"})
    assert result["request_usage"]["reported_prompt_tokens"] == 100
    assert result["token_diagnostics"]["estimated_only"]
    assert result["token_diagnostics"]["projection_estimated_tokens_saved"] == 30
    assert result["token_diagnostics"]["reported_cached_prompt_tokens"] == 20
    assert result["official_correct"] is None and not result["autonomous_complete"]
