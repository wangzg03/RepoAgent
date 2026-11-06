"""Behavioral development cases: no paid models, benchmark solutions or hidden tests."""

import copy
import json
from types import SimpleNamespace

import pytest
from test_harness import ScriptedProvider, history
from test_loop import IMAGE, PREFIX, create_case
from test_native_upstream import local_provider, make_agent, response

from corecoder_mini.accounting import usage_charge
from corecoder_mini.budget import Budget, BudgetExceeded
from corecoder_mini.comparison import compare
from corecoder_mini.context_view import ContextView, TrajectoryTool, message_groups
from corecoder_mini.delivery import digest, export_artifact, read_json
from corecoder_mini.engine import SUBMIT, build_agent
from corecoder_mini.events import Events
from corecoder_mini.progress import (
    ProgressStalled,
    WorkspaceProgress,
    check_command,
    repository_read,
)
from corecoder_mini.reference_runtime import (
    compact_observation,
    episode_index,
    stage_guidance,
)
from corecoder_mini.repair_lab import stage
from corecoder_mini.runtime import local
from corecoder_mini.state_tools import TodoTool, WorkingState
from corecoder_mini.supervisor import (
    DEFAULTS,
    model_descriptor,
    validate_extensions,
    verify_artifact,
)
from corecoder_mini.task_state import atomic_json


@pytest.fixture(autouse=True)
def runtime():
    vars(local).clear()
    yield
    vars(local).clear()


def test_cache_aware_settlement_releases_only_observed_discount():
    rates = (2, 8, .2)
    budget = Budget(.1, token_limit=100000)
    params = {"messages": [], "max_tokens": 100}
    quote = budget.reserve(params, rates)
    assert float(quote) == pytest.approx(.009024)  # Full uncached UTF-8 bound, not a predicted hit.
    # Same observed usage and concrete known cache receipts, matching the audited arithmetic.
    charge = usage_charge({"prompt_tokens": 472883, "completion_tokens": 11061,
                           "prompt_cache_hit_tokens": 438272}, rates)
    assert charge["cost_usd"] == pytest.approx(.2453644)
    observed = budget.reserve(params, rates)
    budget.settle(observed, SimpleNamespace(prompt_tokens=1000, completion_tokens=10,
                  prompt_tokens_details=SimpleNamespace(cached_tokens=900)), rates)
    assert budget.spent == pytest.approx(.00046)
    assert observed.receipt["cost_basis"] == "cache_aware_tariff_estimate"
    assert observed.receipt["cached_prompt_tokens"] == 900
    assert budget.cached_prompt_tokens == 900 and not observed.receipt["is_invoice"]
    # An interrupted transport still consumes its full conservative bound.
    budget.settle(quote, None, rates)
    assert budget.unknown_usage_requests == 1
    assert quote.receipt["cost_basis"] == "unknown_usage_reservation"


@pytest.mark.parametrize("cached", [None, -1, 101, True, "90"])
def test_invalid_cache_never_invents_a_discount(cached):
    charge = usage_charge({"prompt_tokens": 100, "completion_tokens": 10,
                           "prompt_tokens_details": {"cached_tokens": cached}}, (2, 8, .2))
    assert charge["cost_usd"] == pytest.approx(.00028)
    assert charge["cached_prompt_tokens"] is None and charge["cost_basis"] == "uncached_upper_estimate"


def test_cache_tariff_config_is_explicit_and_old_two_rate_still_works():
    descriptor = model_descriptor({"provider": "openai", "model": "fixture", "base_url": "https://example.test/v1",
                                   "pricing_usd_per_million": [.3, 1.2, .03]}, DEFAULTS)
    assert descriptor["pricing_usd_per_million"] == [.3, 1.2, .03]
    charge = usage_charge({"prompt_tokens": 100, "completion_tokens": 10, "prompt_cache_hit_tokens": 90}, (2, 8))
    assert charge["cost_usd"] == pytest.approx(.00028)
    with pytest.raises(BudgetExceeded):
        Budget.quote({"messages": [], "max_tokens": 10}, (float("nan"), 1))


def test_actual_native_transport_uses_cache_tariff_and_missing_usage_stays_unknown(tmp_path, monkeypatch):
    def reply(data, number):
        body = response([("bash", {"command": "echo observed"})])
        body["usage"]["prompt_tokens_details"] = {"cached_tokens": 40}
        return 200, body
    with local_provider(reply) as (url, requests):
        agent = make_agent(tmp_path, monkeypatch, url)
        agent.model.provider.pricing = (1, 2, .1)
        message = agent.model.query([{"role": "user", "content": "Local fixture"}])
        assert len(requests) == 1
        assert message["extra"]["cost"] == pytest.approx(.000054)
        receipt = read_json(tmp_path / "agent/requests/0001.json")
        assert receipt["cached_prompt_tokens"] == 40 and receipt["cost_basis"] == "cache_aware_tariff_estimate"


def test_reused_upstream_compactor_and_real_recreate_error_index(tmp_path):
    text = "line\n" * 1000 + "tail"
    assert "previous line repeated" in compact_observation(text, 1000)
    assert "Economy phase" in stage_guidance(20)["guidance"]
    messages = history(2)
    messages[3]["content"] = "Source example: ValueError: not an execution error"
    messages[5]["content"] = "Traceback\nValueError: actual failure"
    _pinned, groups = message_groups(messages)
    index = episode_index(tmp_path, groups)
    assert [r["step"] for r in index["errors"]] == [2]
    assert index["errors"][0]["type"] == "ValueError" and "expected" not in index


def intent():
    return {"kind": "fix", "summary": "Fix addition for all signs", "constraints": ["Preserve input types"],
            "requirements": [{"id": "positive", "behavior": "Positive sums"}, {"id": "negative", "behavior": "Negative sums"}]}


def test_full_request_and_coverage_survive_restart_without_todo_becoming_proof(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "calc.py").write_text("v1")
    state = WorkingState(tmp_path / "agent")
    full = "User request " + "x" * 5000 + " critical final requirement"
    state.bind_goal(full)
    progress = WorkspaceProgress(workspace, tmp_path / "agent", state=state)
    tool = TodoTool(state, progress)
    tool.execute(intent=intent(), tasks=[{"id": "repair", "content": "Fix and check", "status": "done",
                                         "requirements": ["positive"], "evidence": ["action:1"]}])
    row = {"status": "executed", "result": {"returncode": 0}, "workspace_sha256": progress.version(),
           "verification_candidate": True}
    atomic_json(tmp_path / "agent/actions/0001.json", row)
    report = json.loads(tool.execute(action="check"))
    assert report["uncovered_requirements"] == ["negative"] and report["acceptance"] is False
    assert report["requirements"][0]["current_successful_check_candidates"] == ["action:1"]
    (workspace / "calc.py").write_text("v2")
    assert json.loads(tool.execute(action="check"))["requirements"][0]["current_successful_check_candidates"] == []
    restored = WorkingState(tmp_path / "agent")
    assert restored.data["intent"] == intent()
    page = json.loads(TrajectoryTool(tmp_path / "agent").execute(action="request", offset=4900, limit=1000))
    assert "critical final requirement" in page["content"]
    with pytest.raises(ValueError, match="different_task"):
        restored.bind_goal("Different task")


def test_stall_window_is_durable_and_notes_reads_do_not_replenish_it(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    settings = {"stall_request_limit": 2, "repair_window_requests": 2}
    state = WorkingState(tmp_path / "agent")
    progress = WorkspaceProgress(workspace, tmp_path / "agent", settings, state)
    assert progress.request_feedback() is None
    assert progress.request_feedback()["diagnosis"] == "exploration_stall"
    TodoTool(state).execute(reflection={"failed_approach": "Broad browsing", "hypothesis": "Inspect operator", "next_action": "Run assertion"})
    progress.observe(1, "todo_write", {}, {"returncode": 0})
    assert progress.request_feedback()["remaining_repair_requests"] == 1
    restored = WorkspaceProgress(workspace, tmp_path / "agent", settings, state)
    assert restored.request_feedback()["phase"] == "reanchor"
    restored.observe(2, "scratchpad", {}, {"returncode": 0})
    assert restored.request_feedback()["remaining_repair_requests"] == 1
    restored = WorkspaceProgress(workspace, tmp_path / "agent", settings, state)
    with pytest.raises(ProgressStalled):
        restored.request_feedback()
    assert restored.data["recovery"]["stop_reason"] == "progress_stalled_after_repair_window"


def test_real_change_and_new_check_advance_but_prints_repeated_checks_do_not(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "calc.py"
    source.write_text("v1")
    progress = WorkspaceProgress(workspace, tmp_path / "agent", {"stall_request_limit": 2, "repair_window_requests": 2,
                                                              "max_repair_windows": 4})
    progress.request_feedback()
    progress.request_feedback()
    source.write_text("v2")
    progress.observe(1, "bash", {"command": "edit"}, {"returncode": 0})
    assert progress.request_feedback() is None
    progress.observe(2, "bash", {"command": "python -c 'print(42)'"}, {"returncode": 0})
    assert progress.request_feedback() is not None
    check = "python -c 'assert 1 == 1'"
    progress.observe(3, "bash", {"command": check}, {"returncode": 0, "output": "1 s"})
    assert progress.request_feedback() is None
    progress.observe(4, "bash", {"command": check}, {"returncode": 0, "output": "2 s"})
    assert progress.request_feedback() is not None


@pytest.mark.parametrize("command", ["python inspect.py", "sed -i s/a/b/ x", "cat x > y", "cat /etc/hosts", "ls; touch x", "cat ../x"])
def test_only_pure_scoped_reads_can_be_reused(command):
    assert not repository_read(command)


@pytest.mark.parametrize("command", ["grep -n 'assert x' tests/test_x.py", "cat pytest_helpers.py",
                                     "echo 'assert False'", "python -c 'print(\"assert x\")'",
                                     "python -O -c 'assert False'"])
def test_reads_and_printed_assertions_cannot_renew_repair_window(command):
    assert not check_command(command)


def test_real_assertion_script_is_a_check_candidate_and_outside_script_is_not(tmp_path):
    (tmp_path / "probe.py").write_text("assert 1 == 2\n")
    assert check_command("python probe.py", tmp_path)
    assert check_command("python -m pytest tests/test_x.py")
    assert not check_command("python ../outside.py", tmp_path)


def test_compaction_preserves_current_edit_check_failure_and_opaque_reasoning(tmp_path):
    messages = history(24)
    messages[2 + 2 * 4 + 1]["extra"]["workspace_changed"] = True
    messages[2 + 2 * 5 + 1]["extra"]["verification_candidate"] = True
    original = copy.deepcopy(messages)
    state = WorkingState(tmp_path)
    TodoTool(state).execute(intent=intent(), tasks=[{"content": "Cover edge cases", "status": "pending"}])
    view = ContextView(tmp_path, state, Events(tmp_path), target_tokens=6500, keep_recent=4, history_output_chars=1000)
    projected, receipt = view.project(messages, 12000)
    assert {2, 5, 6, 21, 22, 23, 24} <= set(receipt["retained_turns"])
    assert messages == original and receipt["estimated_tokens_saved"] > 0
    _, groups = message_groups(projected)
    for group in groups:
        assert group[0]["reasoning_content"] == "opaque"
    assert read_json(tmp_path / "private/receipt-index.json")


def fake_run(directory, task_id, prompt, correct=True, cached=0):
    """Fabricated receipts exercise gate logic; these are explicitly NOT official results."""
    directory.mkdir()
    (directory / "model.patch").write_bytes(b"public fixture patch")
    atomic_json(directory / "verification-config.json", {"kind": "swebench_official", "fixture_only": True})
    atomic_json(directory / "frozen.json", {"task_id": task_id, "public_task": "Public fixture " + task_id,
        "image_id": "fixture-image", "model": {"name": "fixture", "pricing_usd_per_million": [1, 2, .1]},
        "settings": DEFAULTS, "baseline_sha256": {"calc.py": "fixture"},
        "source_sha256": {"scripts/swebench_official.py": "unchanged"},
        "verification_config_sha256": digest(directory / "verification-config.json")})
    atomic_json(directory / "agent/submission.json", {"model_submitted": True})
    atomic_json(directory / "official-report.json", {task_id: {"resolved": correct, "fixture_only": True}})
    atomic_json(directory / "verification.json", {"ok": True, "resolved": correct, "patch_applied": True,
                                                 "official_report_sha256": digest(directory / "official-report.json"),
                                                 "patch_sha256": digest(directory / "model.patch")})
    atomic_json(directory / "result.json", {"task_id": task_id, "verification_kind": "swebench_official",
        "patch_sha256": digest(directory / "model.patch"), "evidence_path": str(directory / "verification.json"),
        "evidence_sha256": digest(directory / "verification.json"), "verification_ok": True,
        "artifact_correct": correct, "official_correct": correct, "autonomous_complete": correct, "model_submitted": True})
    atomic_json(directory / "agent/requests/0001.json", {"transport_started": True, "usage_complete": True,
        "prompt_tokens": prompt, "completion_tokens": 100, "cached_prompt_tokens": cached})
    return directory


def test_cost_target_requires_paired_quality_and_actual_cache_usage(tmp_path):
    old = fake_run(tmp_path / "old", "a", 10000, cached=9000)
    new = fake_run(tmp_path / "new", "a", 4000, cached=3600)
    report = compare([old], [new])
    assert report["meets_observed_target"] and report["observed_cost_reduction"] >= .4
    assert not report["policy_promoted"] and not report["is_invoice"]
    atomic_json(new / "agent/requests/0002.json", {"transport_started": True, "usage_complete": False})
    assert "incomplete_usage: a" in compare([old], [new])["reasons"]


def test_equal_aggregate_score_cannot_hide_per_task_regression(tmp_path):
    old = [fake_run(tmp_path / "old-a", "a", 10000), fake_run(tmp_path / "old-b", "b", 10000, False)]
    new = [fake_run(tmp_path / "new-a", "a", 1000, False), fake_run(tmp_path / "new-b", "b", 1000)]
    report = compare(old, new)
    assert report["baseline_official_passed"] == report["candidate_official_passed"] == 1
    assert "per_task_quality_regression: a" in report["reasons"] and not report["meets_observed_target"]
    with pytest.raises(ValueError, match="task_sets"):
        compare(old, new[:1])


def test_zero_success_cannot_be_promoted_and_changed_budget_invalidates_pair(tmp_path):
    old, new = fake_run(tmp_path / "old", "a", 10000, False), fake_run(tmp_path / "new", "a", 1000, False)
    assert "baseline_has_no_autonomous_success" in compare([old], [new])["reasons"]
    frozen = read_json(new / "frozen.json")
    frozen["settings"] = {**DEFAULTS, "token_limit": 1}
    atomic_json(new / "frozen.json", frozen)
    with pytest.raises(ValueError, match="unmatched_model_budget"):
        compare([old], [new])


def test_comparison_rejects_self_reported_submission_and_missing_official_receipt(tmp_path):
    old, new = fake_run(tmp_path / "old", "a", 10000), fake_run(tmp_path / "new", "a", 1000)
    (new / "agent/submission.json").unlink()
    with pytest.raises(ValueError, match="submission_and_completion"):
        compare([old], [new])
    atomic_json(new / "agent/submission.json", {"model_submitted": True})
    (new / "official-report.json").unlink()
    with pytest.raises(ValueError, match="original_official_report"):
        compare([old], [new])


def test_two_rate_legacy_estimates_cannot_prove_actual_cache_aware_savings(tmp_path):
    old, new = fake_run(tmp_path / "old", "a", 10000), fake_run(tmp_path / "new", "a", 1000)
    for directory in (old, new):
        frozen = read_json(directory / "frozen.json")
        frozen["model"]["pricing_usd_per_million"] = [1, 2]
        atomic_json(directory / "frozen.json", frozen)
    report = compare([old], [new])
    assert report["observed_cost_reduction"] > .4 and not report["meets_observed_target"]
    assert "explicit_cache_tariff_required: a" in report["reasons"]


@pytest.mark.parametrize("settings", [{"stall_request_limit": True}, {"repair_window_requests": 0}, {"max_repair_windows": 99}])
def test_invalid_repair_settings_rejected(settings):
    with pytest.raises(ValueError):
        validate_extensions(settings)


@pytest.mark.docker
def test_redundant_read_is_not_executed_and_new_workspace_version_invalidates_receipt(tmp_path):
    create_case(tmp_path)
    commands = [("bash", {"command": "cat calc.py"}), ("bash", {"command": "cat calc.py"}),
                ("bash", {"command": "sed -i 's/a - b/a + b/' calc.py"}),
                ("bash", {"command": "cat calc.py"}), ("bash", {"command": SUBMIT})]
    agent = build_agent(ScriptedProvider(commands), tmp_path / "workspace", tmp_path / "agent", IMAGE, DEFAULTS)
    assert agent.run("Fix addition")["exit_status"] == "Submitted"
    duplicate = read_json(tmp_path / "agent/actions/0002.json")
    assert duplicate["status"] == "reused" and duplicate["result"]["reused_from"] == 1
    assert read_json(tmp_path / "agent/actions/0004.json")["status"] == "executed"


@pytest.mark.docker
def test_automatic_failed_check_reflection_repair_and_verified_delivery(tmp_path):
    create_case(tmp_path)
    check = "python -c 'from calc import add; assert add(2,3)==5; assert add(-1,1)==0'"
    reflection = {"failed_approach": "Repeated reading did not fix the operator", "hypothesis": "Subtraction is wrong", "next_action": "Replace then check both signs"}
    steps = [("todo_write", {"intent": intent(), "tasks": [
        {"id": "fix", "content": "Fix and check signs", "status": "in_progress", "requirements": ["positive", "negative"]}]}),
        ("bash", {"command": check}), ("bash", {"command": "cat calc.py"}),
        ("bash", {"command": "cat calc.py"}), ("todo_write", {"reflection": reflection}),
        ("bash", {"command": "sed -i 's/a - b/a + b/' calc.py"}), ("bash", {"command": check}),
        ("todo_write", {"tasks": [{"id": "fix", "content": "Fix and check signs", "status": "done",
          "requirements": ["positive", "negative"], "evidence": ["action:7"]}]}),
        ("todo_write", {"action": "check"}), ("bash", {"command": SUBMIT})]
    class ReflectionFixture(ScriptedProvider):
        def chat(self, messages, tools):
            if len(self.requests) == 4:
                assert any("Bounded repair feedback" in str(m.get('content')) for m in messages)
                assert any('repair_failure' in str(m.get('content')) for m in messages)
                assert 'Harness next-action guidance' in messages[-1]['content']
                assert 'repair_failure' in messages[-1]['content']
            return super().chat(messages, tools)
    provider = ReflectionFixture(steps)
    agent = build_agent(provider, tmp_path / "workspace", tmp_path / "agent", IMAGE,
                        {**DEFAULTS, "shell_prefix": PREFIX, "stall_request_limit": 3, "repair_window_requests": 5})
    assert agent.run("Fix addition for positive and negative numbers")["exit_status"] == "Submitted"
    coverage = TodoTool(agent.working_state, agent.env.progress).check()
    assert all(r["current_successful_check_candidates"] == ["action:7"] for r in coverage["requirements"])
    export_artifact(tmp_path, {"reason": "fixture"})
    result = verify_artifact(tmp_path)
    assert result["autonomous_complete"] and result["official_correct"] is None
    assert result["task_coverage"]["reflection_recorded"]


def test_stop_before_more_model_spending_exports_incomplete(tmp_path):
    create_case(tmp_path)
    provider = ScriptedProvider([("scratchpad", {"action": "write", "key": "note", "value": "More analysis"})] * 20)
    agent = build_agent(provider, tmp_path / "workspace", tmp_path / "agent", IMAGE,
                        {**DEFAULTS, "stall_request_limit": 2, "repair_window_requests": 2})
    with pytest.raises(ProgressStalled):
        agent.run("Implement addition")
    assert len(provider.requests) == 5 and not (tmp_path / "agent/submission.json").exists()
    assert len(agent.env.progress.data["recovery"]["reanchors"]) == 1
    result = export_artifact(tmp_path, {"reason": "stalled"})
    assert not result["autonomous_complete"] and result["patch_bytes"] == 0
    assert result["workspace_progress"]["recovery"]["stop_reason"] == "progress_stalled_after_repair_window"


@pytest.mark.docker
def test_failed_public_submission_check_returns_to_model_without_restart(tmp_path):
    create_case(tmp_path)
    check = "python -c 'from calc import add; assert add(2,3)==5'"
    class DeliveryFixture(ScriptedProvider):
        def chat(self, messages, tools):
            if len(self.requests) == 1:
                observation = next(m for m in reversed(messages) if m['role'] == 'tool')
                assert "Submission deferred" in observation["content"]
                assert "AssertionError" in observation["content"]
                assert 'repair_failure' in messages[-1]['content']
            return super().chat(messages, tools)
    provider = DeliveryFixture([("bash", {"command": SUBMIT}),
                                ("bash", {"command": "sed -i 's/a - b/a + b/' calc.py"}),
                                ("bash", {"command": SUBMIT})])
    agent = build_agent(provider, tmp_path / "workspace", tmp_path / "agent", IMAGE,
                        {**DEFAULTS, "shell_prefix": PREFIX, "public_check_command": check})
    assert agent.run("Fix addition")["exit_status"] == "Submitted"
    assert len(provider.requests) == 3 and agent.env.actions == 5
    assert read_json(tmp_path / "agent/actions/0001.json")["status"] == "submission_rejected"
    assert read_json(tmp_path / "agent/actions/0002.json")["result"]["returncode"] == 1
    assert read_json(tmp_path / "agent/actions/0005.json")["result"]["returncode"] == 0
    export_artifact(tmp_path, {"reason": "fixture"})
    assert verify_artifact(tmp_path)["autonomous_complete"]


@pytest.mark.docker
def test_child_full_request_review_main_edit_and_own_verification(tmp_path):
    create_case(tmp_path)
    goal = "Fix addition and preserve both signs. " + "context " * 400 + "END_REQUIREMENT"
    class ChildFixture(ScriptedProvider):
        def chat(self, messages, tools):
            names = {t["function"]["name"] for t in tools}
            assert "subagent" not in names and "subagent_report" in names
            if len(self.requests) == 1:
                assert "END_REQUIREMENT" in json.dumps(messages)
            return super().chat(messages, tools)
    def factory(identity, budget):
        return ChildFixture([("trajectory_read", {"action": "request", "offset": 2400, "limit": 2000}),
                             ("bash", {"command": "cat calc.py"}),
                             ("subagent_report", {"summary": "Operator subtracts; check negative and positive inputs.",
                              "findings": [{"claim": "calc.py uses subtraction", "action": 2}]})])
    check = "python -c 'from calc import add; assert add(2,3)==5; assert add(-1,1)==0'"
    steps = [("todo_write", {"intent": intent(), "tasks": [{"id": "repair", "content": "Fix and check", "status": "in_progress",
                                                                  "requirements": ["positive", "negative"]}]}),
             ("subagent", {"action": "spawn", "jobs": [{"id": "review", "task": "Review source and propose checks",
                 "kind": "trajectory_review", "requirements": ["positive", "negative"], "expected_output": "Operator location and sign checks"}]}),
             ("subagent", {"action": "review", "identity": "review", "disposition": "used", "reason": "Use calc.py subtraction finding to change operator, then verify both signs"}),
             ("bash", {"command": "sed -i 's/a - b/a + b/' calc.py"}), ("bash", {"command": check}),
             ("todo_write", {"tasks": [{"id": "repair", "content": "Fix and check", "status": "done", "requirements": ["positive", "negative"], "evidence": ["action:5"]}]}),
             ("todo_write", {"action": "check"}), ("bash", {"command": SUBMIT})]
    agent = build_agent(ScriptedProvider(steps), tmp_path / "workspace", tmp_path / "agent", IMAGE,
                        {**DEFAULTS, "shell_prefix": PREFIX}, child_provider_factory=factory)
    assert agent.run(goal)["exit_status"] == "Submitted"
    child = read_json(tmp_path / "agent/children.json")["review"]
    assert child["status"] == "completed" and child["review"]["source_actions"] == [2]
    assert child["input"]["overall_goal_truncated"] and child["changed_paths"] == []
    assert child["input"]["requirements"] == intent()["requirements"]
    export_artifact(tmp_path, {"reason": "fixture"})
    result = verify_artifact(tmp_path)
    assert result["autonomous_complete"] and result["harness_observation"]["used_children"] == ["review"]


def test_recreate_outer_candidate_is_versioned_isolated_and_rejects_benchmark(tmp_path):
    create_case(tmp_path)
    (tmp_path / "workspace/calc.py").write_text("def add(a,b):\n    return a+b\n")
    atomic_json(tmp_path / "agent/submission.json", {"model_submitted": True})
    result = export_artifact(tmp_path, {"reason": "fixture"})
    evidence = tmp_path / "verification.json"
    atomic_json(evidence, {"ok": True, "resolved": True, "patch_applied": True, "patch_sha256": result["patch_sha256"]})
    from corecoder_mini.delivery import bind_verdict
    bind_verdict(tmp_path, read_json(evidence), evidence, kind="local_command")
    atomic_json(tmp_path / "agent/actions/0001.json", {"status": "executed", "result": {"returncode": 1}})
    atomic_json(tmp_path / "agent/trajectory.json", {"info": {"config": {"agent": {
        "system_template": "Public development instructions", "instance_template": "Task {{task}}"}}}})
    before = digest(tmp_path / "workspace/calc.py")
    destination = tmp_path.parent / (tmp_path.name + "-candidate")
    report = stage(tmp_path, destination, {"reason": "Use observed failure to narrow next check", "evidence_actions": [1],
                                          "changes": {"instance_template": "修复当前任务 {{task}}，检查全部边界"}})
    import yaml
    current = yaml.safe_load((destination / "scaffold/current_scaffold.yaml").read_text(encoding="utf-8"))
    assert current["version"] == 2 and "全部边界" in current["instance_template"]
    assert (destination / "scaffold/history/v0001.yaml").exists()
    assert not report["active"] and not report["automatic_promotion"]
    assert digest(tmp_path / "workspace/calc.py") == before
    frozen = read_json(tmp_path / "frozen.json")
    frozen["settings"] = {**DEFAULTS, "task_kind": "benchmark"}
    atomic_json(tmp_path / "frozen.json", frozen)
    with pytest.raises(ValueError, match="benchmark_episode"):
        stage(tmp_path, tmp_path.parent / (tmp_path.name + "-forbidden"), {"changes": {"step_limit": 999}})
