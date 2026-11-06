"""Public development fixtures; no benchmark answers and no model calls."""

import copy
import json
from typing import ClassVar

import pytest
from test_harness import ScriptedProvider, history
from test_loop import IMAGE, PREFIX, create_case

from corecoder_mini.context import estimate_tokens
from corecoder_mini.context_view import ContextView, message_groups
from corecoder_mini.delivery import export_artifact, read_json
from corecoder_mini.engine import SUBMIT, ProviderModel, build_agent, templates
from corecoder_mini.events import Events
from corecoder_mini.llm import LLMResponse, ToolCall
from corecoder_mini.progress import WorkspaceProgress
from corecoder_mini.runtime import local
from corecoder_mini.state_tools import WorkingState
from corecoder_mini.supervisor import DEFAULTS, verify_artifact
from corecoder_mini.tools.base import Tool
from corecoder_mini.tools.registry import ToolRegistry
from corecoder_mini.upstream import FormatError


@pytest.fixture(autouse=True)
def clear_runtime():
    vars(local).clear()
    yield
    vars(local).clear()


@pytest.mark.docker
def test_extension_turn_needs_no_bash_and_baseline_prompt_is_unchanged(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    llm = ScriptedProvider([("scratchpad", {"action": "write", "key": "hypothesis", "value": "Inspect arithmetic"}),
                            ("bash", {"command": SUBMIT})])
    agent = build_agent(llm, workspace, tmp_path / "agent", IMAGE, DEFAULTS)
    assert "AT LEAST ONE bash" not in agent.config.instance_template
    assert agent.run("Read-only investigation fixture")["exit_status"] == "Submitted"
    assert agent.env.actions == 2 and read_json(tmp_path / "agent/actions/0001.json")["tool"] == "scratchpad"
    plain = build_agent(ScriptedProvider([]), workspace, tmp_path / "plain", IMAGE,
                        {**DEFAULTS, "harness_tools": []})
    assert plain.config.instance_template == templates()["agent"]["instance_template"]
    assert plain.model.view is None and plain.env.progress is None


def test_mixed_submission_rejected_before_any_action(tmp_path):
    class Mixed(ScriptedProvider):
        def chat(self, messages, tools):
            return LLMResponse(tool_calls=[ToolCall("one", "bash", {"command": "touch premature"}),
                                           ToolCall("two", "bash", {"command": SUBMIT})], prompt_tokens=10)
    model = ProviderModel(Mixed([]), tmp_path)
    with pytest.raises(FormatError) as exc:
        model.query([{"role": "user", "content": "Implement a change"}])
    assert "only tool call" in exc.value.messages[0]["content"]
    assert not (tmp_path / "actions").exists()


@pytest.mark.docker
def test_installed_extension_uses_extension_contract_without_builtin_helpers(tmp_path, monkeypatch):
    class Inspect(Tool):
        name = "inspect_fixture"
        description = "Inspect a public fixture"
        parameters: ClassVar[dict] = {"type": "object", "properties": {}, "additionalProperties": False}
        def execute(self):
            return "Public fixture inspection complete"
    monkeypatch.setattr(ToolRegistry, "load_extensions", lambda registry, names: registry.register(Inspect()))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provider = ScriptedProvider([("inspect_fixture", {}), ("bash", {"command": SUBMIT})])
    agent = build_agent(provider, workspace, tmp_path / "agent", IMAGE,
                        {**DEFAULTS, "harness_tools": [], "tool_extensions": ["installed-fixture"]})
    assert "AT LEAST ONE bash" not in agent.config.instance_template
    assert agent.model.view is not None
    assert agent.run("Inspect only")["exit_status"] == "Submitted"


def test_schema_error_explains_repair_and_keeps_extensions_available(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provider = ScriptedProvider([("scratchpad", {"action": "not_an_action"})])
    agent = build_agent(provider, workspace, tmp_path / "agent", IMAGE, DEFAULTS)
    with pytest.raises(FormatError) as exc:
        agent.model.query([{"role": "user", "content": "Inspect"}])
    text = exc.value.messages[0]["content"]
    assert "scratchpad" in text and "bash remains available" in text
    assert "AT LEAST ONE bash" not in agent.model.config["format_error_template"]


def test_progress_reads_notes_and_different_success_do_not_erase_failure(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    file = workspace / "calc.py"
    file.write_text("old")
    observer = WorkspaceProgress(workspace, tmp_path / "agent")
    observer.observe(1, "bash", {"command": "python check.py"}, {"returncode": 1})
    for action in range(2, 8):
        observer.observe(action, "bash", {"command": f"cat file{action}.py"}, {"returncode": 0})
    observer.observe(8, "todo_write", {}, {"returncode": 0})
    restored = WorkspaceProgress(workspace, tmp_path / "agent")
    feedback = restored.feedback()
    assert feedback["actions_without_workspace_change"] == 8
    assert feedback["workspace_changed_files"] == 0 and "next_action_guidance" in feedback
    assert feedback["recent_nonzero_commands"][0]["action_id"] == 1
    file.write_text("new")
    restored.observe(9, "bash", {"command": "edit fixture"}, {"returncode": 0})
    assert restored.feedback()["workspace_changed_files"] == 1
    assert restored.feedback()["actions_without_workspace_change"] == 0
    assert restored.feedback()["recent_nonzero_commands"]  # Editing is not verification.
    restored.observe(10, "bash", {"command": "python check.py"}, {"returncode": 0})
    assert restored.feedback()["recent_nonzero_commands"] == []
    file.write_text("old")
    restored.observe(11, "bash", {"command": "restore fixture"}, {"returncode": 0})
    assert restored.feedback()["workspace_changed_files"] == 0


def test_progress_observation_error_is_unknown_not_a_false_change(tmp_path, monkeypatch):
    observer = WorkspaceProgress(tmp_path, tmp_path.parent / (tmp_path.name + "-state"))
    def unavailable(*args):
        raise OSError("fixture")
    monkeypatch.setattr("corecoder_mini.progress.tree_hashes", unavailable)
    observer.observe(1, "bash", {"command": "read"}, {"returncode": 0})
    assert observer.feedback()["observation_error"] == "OSError"
    assert observer.feedback()["workspace_changed_files"] == 0


def small_history(count):
    messages = history(count)
    for message in messages:
        if message["role"] == "tool":
            message["content"] = "observation " * 120
    return messages


def test_checkpoint_prefix_survives_append_and_restart_without_rewriting_old_output(tmp_path):
    state, events = WorkingState(tmp_path), Events(tmp_path)
    view = ContextView(tmp_path, state, events, target_tokens=4000, keep_recent=2, history_output_chars=512)
    messages = small_history(30)
    original = copy.deepcopy(messages)
    first, receipt = view.project(messages, 10000)
    assert receipt["compacted"]
    assert 2 in receipt["retained_turns"]  # The actual failed observation survives.
    assert messages == original
    view = ContextView(tmp_path, state, events, target_tokens=4000, keep_recent=2, history_output_chars=512)
    for count in (31, 32):
        messages = small_history(count)
        second, next_receipt = view.project(messages, 10000)
        assert next_receipt["checkpoint_reused"]
        assert next_receipt["checkpoint_generation"] == receipt["checkpoint_generation"]
        assert second[:len(first) - 1] == first[:-1]  # Only the live memory suffix is replaced.
        message_groups(second)
        first = second
    assert read_json(tmp_path / "canonical-messages.json") == messages


def test_checkpoint_invalidated_on_changed_canonical_history_and_capacity_shrink(tmp_path):
    state, events = WorkingState(tmp_path), Events(tmp_path)
    view = ContextView(tmp_path, state, events, target_tokens=4000, keep_recent=2)
    messages = small_history(30)
    _first, receipt = view.project(messages, 10000)
    messages[1]["content"] = "Changed task constraint"
    second, updated = view.project(messages, 10000)
    assert not updated["checkpoint_reused"] and "Changed task constraint" in json.dumps(second)
    assert updated["checkpoint_generation"] > receipt["checkpoint_generation"]
    reduced, update = view.project(small_history(31), capacity=3200)
    assert estimate_tokens(reduced) <= 3200 and 2 in update["retained_turns"]
    with pytest.raises(RuntimeError, match="cannot fit safely"):
        view.project(small_history(32), capacity=100)


@pytest.mark.docker
def test_feedback_drives_fixture_from_reading_to_edit_check_and_bound_delivery(tmp_path):
    create_case(tmp_path)
    checks = "python -c 'from calc import add; assert add(2,3)==5; assert add(-1,1)==0'"
    actions = [("bash", {"command": checks})]
    actions += [("bash", {"command": "cat calc.py"})] * 5
    actions += [("bash", {"command": "python -c \"from pathlib import Path; p=Path('calc.py'); p.write_text(p.read_text().replace('a - b','a + b'))\""}),
                ("bash", {"command": checks}), ("bash", {"command": SUBMIT})]
    class FeedbackFixture(ScriptedProvider):
        def chat(self, messages, tools):
            memory = next(m['content'] for m in reversed(messages)
                          if str(m.get('content')).startswith('Current task memory'))
            if len(self.requests) == 6:
                assert '"workspace_changed_files":0' in memory
                assert '"unresolved_check_actions":[1]' in memory
                assert 'repair_failure' in messages[-1]['content']
            if len(self.requests) == 7:
                assert '"workspace_changed_files":1' in memory
                assert '"unresolved_check_actions":[1]' in memory
                assert 'verify_current' in messages[-1]['content']
            if len(self.requests) == 8:
                assert '"unresolved_check_actions":[]' in memory
                assert 'review_delivery' in messages[-1]['content']
            return super().chat(messages, tools)
    provider = FeedbackFixture(actions)
    agent = build_agent(provider, tmp_path / "workspace", tmp_path / "agent", IMAGE,
                        {**DEFAULTS, "shell_prefix": PREFIX})
    assert agent.run("Fix add and verify its behavior")["exit_status"] == "Submitted"
    result = export_artifact(tmp_path, {"reason": "fixture_done"})
    assert result["workspace_progress"]["changed_paths"] == ["calc.py"]
    assert not result["autonomous_complete"]  # Submission alone cannot accept the task.
    final = verify_artifact(tmp_path)
    assert final["autonomous_complete"] and final["official_correct"] is None
