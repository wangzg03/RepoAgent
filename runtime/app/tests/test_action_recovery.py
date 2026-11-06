"""Public toy tasks exercise recovery mechanics, not real model or benchmark quality."""

import copy
import json
import threading

import pytest
from test_loop import create_case
from test_native_upstream import local_provider, make_agent, response

from corecoder_mini.budget import Budget, BudgetExceeded
from corecoder_mini.command_observation import check_command, observation_kind
from corecoder_mini.context_view import ContextView, message_groups
from corecoder_mini.delivery import export_artifact, read_json
from corecoder_mini.engine import SUBMIT
from corecoder_mini.events import Events
from corecoder_mini.progress import ProgressStalled, WorkspaceProgress
from corecoder_mini.runtime import Cancelled, local
from corecoder_mini.state_tools import ScratchpadTool, WorkingState
from corecoder_mini.supervisor import verify_artifact
from corecoder_mini.tools.registry import ToolRegistry


@pytest.fixture(autouse=True)
def clear_runtime():
    vars(local).clear()
    yield
    vars(local).clear()


@pytest.mark.parametrize('command', [
    "cd /workspace && python -c 'assert 1 == 1'",
    "export PYTHONPATH=/workspace; python -m pytest tests/test_calc.py",
    "PYTHONPATH=/workspace python -m unittest",
    "cd /workspace && env X=1 python - <<'PY'\nassert 1 == 1\nPY\n",
    'python - <<"PY"\nfrom calc import add\nassert add(2,3)==5\nPY',
    "python bin/test tests/test_calc.py",
])
def test_wrapped_real_checks_are_recognized(command, tmp_path):
    assert check_command(command, tmp_path)


@pytest.mark.parametrize('command', [
    "cd /workspace && python -c 'print(\"assert False\")'",
    "cd /workspace && grep assert calc.py",
    "python - <<'PY'\nprint('assert False')\nPY",
    "python - <<'PY'\nassert False\nPY\ntrue",
    "python -c 'assert False' || true",
    "python -c 'assert False' | cat",
    "python -O -c 'assert False'",
    "cd /workspace/../outside && python -c 'assert True'",
])
def test_prints_reads_masked_checks_and_outside_paths_are_not_verification(command, tmp_path):
    assert not check_command(command, tmp_path)


def test_relative_script_after_cd_is_scoped(tmp_path):
    (tmp_path / 'pkg').mkdir()
    (tmp_path / 'pkg/probe.py').write_text('assert 1 == 1\n')
    assert check_command('cd /workspace/pkg && python probe.py', tmp_path)
    assert not check_command('cd /workspace/pkg && python ../../outside.py', tmp_path)


def test_diagnostics_cannot_poison_a_real_check_failure(tmp_path):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    progress = WorkspaceProgress(workspace, tmp_path / 'agent')
    first = progress.observe(1, 'bash', {'command': "cd /workspace && ls && git log -1 && python -c 'print(1)';\n\n"},
                             {'returncode': 128, 'output': 'fatal: not a git repository'})
    assert first['diagnostic_kind'] == 'git_history_unavailable_use_source_files'
    second = progress.observe(2, 'bash', {'command': 'cd /workspace && grep absent missing.py'},
                              {'returncode': 1, 'output': ''})
    assert second['diagnostic_kind'] == 'search_no_match'
    assert not progress.data['unresolved_commands']
    check = "cd /workspace && python - <<'PY'\nassert False\nPY"
    assert progress.observe(3, 'bash', {'command': check}, {'returncode': 1, 'output': 'AssertionError'})['verification_candidate']
    assert [r['action_id'] for r in progress.data['unresolved_commands']] == [3]
    progress.observe(4, 'bash', {'command': 'grep other missing.py'}, {'returncode': 0, 'output': 'found'})
    assert [r['action_id'] for r in progress.data['unresolved_commands']] == [3]
    progress.observe(5, 'bash', {'command': check}, {'returncode': 0, 'output': ''})
    assert not progress.data['unresolved_commands']
    assert observation_kind('sed -i s/a/b/ calc.py', tmp_path) == 'execution'


def test_reanchor_then_distinct_investigation_does_not_get_cut_off_like_repetition(tmp_path):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    for i in range(12):
        (workspace / f'module{i}.py').write_text(f'value = {i}\n')
    progress = WorkspaceProgress(workspace, tmp_path / 'agent', {'stall_request_limit': 2, 'repair_window_requests': 2})
    for i in range(10):
        feedback = progress.request_feedback()
        progress.observe(i + 1, 'bash', {'command': f'cat module{i}.py'}, {'returncode': 0, 'output': f'value = {i}'})
    assert feedback['phase'] == 'converge'
    assert len(progress.data['recovery']['reanchors']) == 1
    assert progress.data['productive_epoch'] == 0  # New reads are not edits or checks.
    # Repeating old evidence does not perpetually renew the investigation.
    for i in range(2):
        progress.request_feedback()
        progress.observe(i + 20, 'bash', {'command': 'cat module9.py'}, {'returncode': 0, 'output': 'value = 9'})
    with pytest.raises(ProgressStalled):
        progress.request_feedback()


def test_reanchor_preserves_protocol_task_failure_and_real_source_without_replaying(tmp_path):
    state = WorkingState(tmp_path)
    goal = 'Repair the public toy API. END_REQUIREMENT'
    messages = [{'role': 'system', 'content': 'Public development task'}, {'role': 'user', 'content': goal}]
    for i in range(10):
        messages += [{'role': 'assistant', 'reasoning_content': 'opaque', 'tool_calls': [
            {'id': f'call-{i}', 'type': 'function', 'function': {'name': 'bash', 'arguments': json.dumps({'command': f'cat module{i}.py'})}}]},
            {'role': 'tool', 'tool_call_id': f'call-{i}', 'content': f'ACTUAL_SOURCE_{i}',
             'extra': {'action_id': i + 1, 'returncode': 1 if i == 1 else 0, 'workspace_changed': i == 2,
                       'verification_candidate': i == 3, 'observation_kind': 'inspection', 'source_targets': [f'module{i}.py']}}]
    original = copy.deepcopy(messages)
    view = ContextView(tmp_path, state, Events(tmp_path), recovery=lambda: {'reanchor_id': 1}, target_tokens=8000)
    projected, receipt = view.project(messages, 16000)
    assert receipt['reanchor_id'] == 1 and not receipt['compacted']
    assert receipt['retained_turns'] == list(range(1, 11))
    assert {2, 3, 4, 9, 10} <= set(receipt['retained_turns'])
    assert goal in json.dumps(projected) and 'ACTUAL_SOURCE_' in json.dumps(projected)
    assert messages == original and read_json(tmp_path / 'canonical-messages.json') == original
    assert read_json(tmp_path / 'private/action-recovery/0001.json')['replayed_tools'] == 0
    assert not read_json(tmp_path / 'private/action-recovery/0001.json')['history_pruned']
    _, groups = message_groups(projected)
    assert all(g[0]['reasoning_content'] == 'opaque' for g in groups)
    restored = ContextView(tmp_path, state, Events(tmp_path), recovery=lambda: {'reanchor_id': 1}, target_tokens=8000)
    _again, second = restored.project(messages, 16000)
    assert second['checkpoint_reused'] and second['checkpoint_generation'] == receipt['checkpoint_generation']


def test_scratchpad_contract_validates_before_execution_without_new_schema_dialect(tmp_path):
    state = WorkingState(tmp_path)
    tool = ScratchpadTool(state)
    registry = ToolRegistry([tool])
    for value in ['plain note', [], 1, None]:
        with pytest.raises(ValueError, match='action=write'):
            registry.validate('scratchpad', {'action': 'merge', 'key': 'hypothesis', 'value': value})
    assert state.data['revision'] == 0
    for value in [{'hypothesis': 'Inspect operator'}, '{"hypothesis":"Inspect operator"}']:
        registry.validate('scratchpad', {'action': 'merge', 'value': value})
        tool.execute(action='merge', value=value)
    registry.validate('scratchpad', {'action': 'write', 'key': 'hypothesis', 'value': 'plain text'})
    assert 'if' not in tool.schema()['function']['parameters']


def test_redundant_wrappers_comments_and_spacing_do_not_count_as_new_evidence(tmp_path):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    (workspace / 'calc.py').write_text('value = 1\n')
    progress = WorkspaceProgress(workspace, tmp_path / 'agent')
    for i, command in enumerate(['cat calc.py', 'cd /workspace && cat calc.py', 'cat   calc.py # new plan']):
        progress.observe(i + 1, 'bash', {'command': command}, {'returncode': 0, 'output': 'value = 1'})
    assert progress.data['evidence_epoch'] == 1
    assert observation_kind('find . -delete', tmp_path) == 'execution'


def test_one_ambiguous_exit_does_not_poison_investigation_and_check_failures_are_not_evicted(tmp_path):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    progress = WorkspaceProgress(workspace, tmp_path / 'agent')
    query = "python -c 'import missing_optional_module'"
    progress.observe(1, 'bash', {'command': query}, {'returncode': 1})
    assert progress.data['unresolved_commands'] and not progress._blocking_failures()
    progress.observe(2, 'bash', {'command': query}, {'returncode': 1})
    assert [r['action_id'] for r in progress._blocking_failures()] == [2]
    progress.observe(3, 'bash', {'command': "python -c 'assert False'"}, {'returncode': 1})
    for i in range(8):
        progress.observe(i + 4, 'bash', {'command': f'unknown_program_{i}'}, {'returncode': 1})
    assert 3 in [r['action_id'] for r in progress._blocking_failures()]
    assert len(progress.data['unresolved_commands']) == 10


def test_recovery_does_not_bypass_cancel_or_shared_budget(tmp_path, monkeypatch):
    def forbidden(*args):
        pytest.fail('No transport may start after cancellation or failed admission')
    with local_provider(forbidden) as (url, requests):
        agent = make_agent(tmp_path, monkeypatch, url, extensions=['trajectory_read'], budget=Budget(.1, 1, 1))
        with pytest.raises(BudgetExceeded):
            agent.model.query([{'role': 'user', 'content': 'Public task'}])
        local.cancel = threading.Event()
        local.cancel.set()
        with pytest.raises(Cancelled):
            agent.model.query([{'role': 'user', 'content': 'Public task'}])
        assert not requests


@pytest.mark.docker
def test_native_loop_reanchors_after_eight_reads_then_edits_repairs_checks_and_delivers(tmp_path, monkeypatch):
    create_case(tmp_path)
    for i in range(8):
        (tmp_path / 'workspace' / f'context{i}.txt').write_text(f'Public source note {i}\n')
        (tmp_path / 'baseline' / f'context{i}.txt').write_text(f'Public source note {i}\n')
    from scripts.swebench_io import tree_hashes
    frozen = read_json(tmp_path / 'frozen.json')
    frozen['baseline_sha256'] = tree_hashes(tmp_path / 'baseline')
    from corecoder_mini.task_state import atomic_json
    atomic_json(tmp_path / 'frozen.json', frozen)
    check = "cd /workspace && python - <<'PY'\nfrom calc import add\nassert add(2,3)==5\nassert add(-1,1)==0\nPY"
    steps = [('bash', {'command': f'cd /workspace && cat context{i}.txt calc.py'}) for i in range(8)]
    steps += [('bash', {'command': "sed -i 's/a - b/a * b/' calc.py"}), ('bash', {'command': check}),
              ('bash', {'command': r"sed -i 's/a \* b/a + b/' calc.py"}), ('bash', {'command': check}),
              ('bash', {'command': SUBMIT})]
    def answer(data, number):
        if number == 9:
            wire = json.dumps(data['messages'])
            assert 'reanchor_id' in wire and 'recovery_packet' not in wire
            assert read_json(tmp_path / 'agent/workspace-progress.json')['recovery']['reanchors']
            assert 'Fix the public add API' in wire and 'def add(a, b)' in wire
            assert 'opaque-native-state' in wire
            message_groups(data['messages'])
        if number == 11:
            assert 'AssertionError' in json.dumps(data['messages'])
        return 200, response([steps[number - 1]])
    with local_provider(answer) as (url, requests):
        agent = make_agent(tmp_path, monkeypatch, url, extensions=['scratchpad', 'todo_write', 'trajectory_read'],
                           budget=Budget(.1, 300000, 24), context_tokens=32000)
        assert agent.run('Fix the public add API and check both signs.')['exit_status'] == 'Submitted'
        assert len(requests) == 13 and agent.model.provider.budget.requests == 13
    assert read_json(tmp_path / 'agent/private/action-recovery/0001.json')['replayed_tools'] == 0
    assert read_json(tmp_path / 'agent/actions/0010.json')['verification_candidate']
    assert read_json(tmp_path / 'agent/actions/0010.json')['result']['returncode'] == 1
    assert read_json(tmp_path / 'agent/actions/0012.json')['result']['returncode'] == 0
    assert agent.env.progress.data['changed_paths'] == ['calc.py']
    export_artifact(tmp_path, {'reason': 'scripted development fixture'})
    result = verify_artifact(tmp_path)
    assert result['autonomous_complete'] and result['official_correct'] is None
    assert result['patch_bytes'] > 0 and result['model_submitted']
