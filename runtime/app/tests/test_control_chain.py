"""Public development cases for control flow; not benchmark/model accuracy."""

import json
import threading

import pytest
from test_loop import create_case
from test_native_upstream import local_provider, make_agent, response

from corecoder_mini.budget import Budget
from corecoder_mini.command_observation import (
    check_command,
    observation_kind,
    search_result_kind,
)
from corecoder_mini.continuation import ActionContinuation
from corecoder_mini.delivery import export_artifact, read_json
from corecoder_mini.engine import SUBMIT
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
    "grep -n absent calc.py 2>/dev/null",
    "cd /workspace && grep -n absent calc.py 2> /dev/null",
    "find . -name '*.py' -print0 | xargs -0 grep -n absent 2>/dev/null",
    "find . -name '*.py' | xargs grep -n absent 2>/dev/null",
    "grep -n add calc.py | head -n 3",
])
def test_read_only_searches_keep_source_targets_without_becoming_checks(tmp_path, command):
    (tmp_path / 'calc.py').write_text('value = 1\n')
    assert observation_kind(command, tmp_path) == 'search'
    assert not check_command(command, tmp_path)


@pytest.mark.parametrize('command', [
    "find . -exec sh -c 'touch changed' ';' | grep calc",
    "find . -delete | grep calc",
    "find . -name '*.py' | xargs sh -c 'touch changed'",
    "grep calc calc.py > changed.py",
    "grep calc calc.py | sort -o changed.py",
    "uniq calc.py changed.py",
    "sed --in-place s/a/b/ calc.py",
    "rg --pre 'touch changed' calc",
    "python -c 'assert False' | grep calc",
    "grep calc calc.py && python -c 'raise RuntimeError()'",
])
def test_mutating_or_unknown_pipelines_are_never_demoted_to_search(tmp_path, command):
    assert observation_kind(command, tmp_path) == 'execution'
    assert not check_command(command, tmp_path)


@pytest.mark.parametrize('command', [
    "python -c 'assert True' 2>/dev/null",
    "python -m pytest tests/test_calc.py >/dev/null",
])
def test_null_redirect_does_not_hide_an_actual_check(tmp_path, command):
    assert check_command(command, tmp_path)
    assert observation_kind(command, tmp_path) == 'check'


def controller(workspace, directory):
    state = WorkingState(directory)
    state.bind_task_contract('fix')
    progress = WorkspaceProgress(workspace, directory, {'stall_request_limit': 3, 'repair_window_requests': 2}, state)
    return progress, ActionContinuation(progress, type('Registry', (), {'tools': {'bash': None, 'read_file': None, 'edit_file': None}})())


def test_repeated_no_match_survives_restart_and_guides_a_different_action(tmp_path):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    (workspace / 'calc.py').write_text('value = 1\n')
    command = 'grep absent calc.py 2>/dev/null'
    for i in range(3):
        progress, continuation = controller(workspace, tmp_path / 'agent')
        progress.request_feedback()
        receipt = progress.observe(i + 1, 'bash', {'command': command}, {'returncode': 1, 'output': ''})
        assert receipt['diagnostic_kind'] == 'search_no_match'
        packet = continuation()
        assert packet['status'] == 'change_search_strategy'
        assert 'not a failed test' in packet['instruction'] and 'Do not rerun' in packet['instruction']
        assert packet['next_action_hint'] == {'tool': 'read_file', 'arguments': {'file_path': 'calc.py'}}
        assert packet['search_observation']['target_paths'] == ['calc.py']
        assert packet['acceptance'] is False  # An existing search operand is not a successful source read.
        assert progress.search_feedback()['repetitions'] == i + 1
        assert not progress._blocking_failures() and not progress.data['unresolved_commands']
    # A targeted read takes the agent out of search recovery, with editing available.
    progress.observe(4, 'read_file', {'file_path': 'calc.py'}, {'returncode': 0})
    assert continuation()['status'] == 'implement_or_discriminate'
    assert continuation()['facts']['located_sources'][0]['path'] == 'calc.py'


@pytest.mark.parametrize('command,code', [
    ('grep absent calc.py 2>/dev/null', 2),
    ("find . -name '*.py' | xargs grep absent", 123),
    ('grep absent calc.py | head', 1),
])
def test_search_error_is_actionable_and_not_mislabeled_no_match(tmp_path, command, code):
    assert search_result_kind(command, code, tmp_path) == 'search_error'
    progress, continuation = controller(tmp_path, tmp_path.parent / (tmp_path.name + '-state'))
    progress.observe(1, 'bash', {'command': command}, {'returncode': code, 'output': ''})
    assert continuation()['status'] == 'change_search_strategy'
    assert 'Correct its arguments or use read_file' in continuation()['instruction']
    assert not progress._blocking_failures()


def test_no_match_never_clears_a_real_failed_check_even_after_source_changes(tmp_path):
    source = tmp_path / 'calc.py'
    source.write_text('value = 1\n')
    progress, continuation = controller(tmp_path, tmp_path.parent / (tmp_path.name + '-state'))
    check = "python -c 'assert False' 2>/dev/null"
    progress.observe(1, 'bash', {'command': check}, {'returncode': 1})
    progress.observe(2, 'bash', {'command': 'grep missing calc.py 2>/dev/null'}, {'returncode': 1})
    assert continuation()['status'] == 'repair_failure'
    source.write_text('value = 2\n')
    progress.observe(3, 'edit_file', {'file_path': 'calc.py'}, {'returncode': 0})
    assert continuation()['status'] == 'verify_current'
    assert len(progress._blocking_failures()) == 1


def test_an_old_unknown_invocation_does_not_require_reexecution_after_actual_progress(tmp_path):
    source = tmp_path / 'calc.py'
    source.write_text('value = 1\n')
    progress, continuation = controller(tmp_path, tmp_path.parent / (tmp_path.name + '-state'))
    for i in range(2):
        progress.observe(i + 1, 'bash', {'command': 'unknown_probe'}, {'returncode': 127})
    assert continuation()['status'] == 'repair_execution'
    source.write_text('value = 2\n')
    progress.observe(3, 'edit_file', {'file_path': 'calc.py'}, {'returncode': 0})
    assert continuation()['status'] == 'verify_current'
    assert progress.data['unresolved_commands']  # Historical diagnostic not erased or called successful.


def test_repeated_searches_still_stop_with_a_truthful_reason(tmp_path):
    progress, continuation = controller(tmp_path, tmp_path.parent / (tmp_path.name + '-state'))
    with pytest.raises(ProgressStalled):
        for i in range(20):
            progress.request_feedback()
            progress.observe(i + 1, 'bash', {'command': 'grep absent calc.py 2>/dev/null'}, {'returncode': 1})
            assert continuation()['status'] == 'change_search_strategy'
    assert progress.data['recovery']['stop_cause'] == 'repeated_observations_or_bookkeeping'
    assert not progress.data['unresolved_commands']


def test_scratchpad_bad_write_is_rejected_before_side_effect_then_can_be_corrected(tmp_path):
    state = WorkingState(tmp_path)
    registry = ToolRegistry([ScratchpadTool(state)])
    with pytest.raises(ValueError, match='requires a nonempty key'):
        registry.validate('scratchpad', {'action': 'write', 'value': 'inspect one exact function'})
    with pytest.raises(ValueError, match='requires value'):
        registry.validate('scratchpad', {'action': 'write', 'key': 'hypothesis'})
    assert not state.data['notes'] and not state.path.exists()
    registry.validate('scratchpad', {'action': 'write', 'key': 'hypothesis', 'value': 'inspect one exact function'})
    registry.tools['scratchpad'].execute('write', 'hypothesis', 'inspect one exact function')
    assert WorkingState(tmp_path).data['notes']['hypothesis'] == 'inspect one exact function'


def test_control_guidance_preserves_cancellation(tmp_path):
    progress, continuation = controller(tmp_path, tmp_path.parent / (tmp_path.name + '-state'))
    progress.observe(1, 'bash', {'command': 'grep absent calc.py 2>/dev/null'}, {'returncode': 1})
    local.cancel = threading.Event()
    local.cancel.set()
    with pytest.raises(Cancelled):
        continuation()


@pytest.mark.docker
def test_native_control_chain_from_search_and_schema_errors_to_edit_test_repair_and_submission(tmp_path, monkeypatch):
    create_case(tmp_path)
    check = "python -c 'from calc import add; assert add(2,3)==5; assert add(-1,1)==0' 2>/dev/null"
    query = 'grep PUBLIC_MISSING_SYMBOL calc.py 2>/dev/null'
    actions = [
        ('bash', {'command': "find . -name '*.py' -print0 | xargs -0 grep PUBLIC_MISSING_SYMBOL 2>/dev/null"}),
        ('bash', {'command': query}), ('bash', {'command': query}),
        ('scratchpad', {'action': 'write', 'value': 'Use the located arithmetic function'}),
        ('scratchpad', {'action': 'write', 'key': 'hypothesis', 'value': 'Use the located arithmetic function'}),
        ('read_file', {'file_path': 'calc.py'}),
        ('edit_file', {'file_path': 'calc.py', 'old_string': 'return a - b', 'new_string': 'return abs(a) + abs(b)'}),
        ('bash', {'command': check}), ('bash', {'command': query}),
        ('edit_file', {'file_path': 'calc.py', 'old_string': 'return abs(a) + abs(b)', 'new_string': 'return a + b'}),
        ('bash', {'command': check}), ('bash', {'command': SUBMIT}),
    ]
    expected = {2: 'change_search_strategy', 3: 'change_search_strategy', 4: 'change_search_strategy',
                7: 'implement_or_discriminate', 8: 'verify_current', 9: 'repair_failure',
                10: 'repair_failure', 11: 'verify_current', 12: 'review_delivery'}
    def reply(data, number):
        if number in expected:
            assert read_json(tmp_path / 'agent/private/action-continuation.json')['status'] == expected[number]
        if number == 5:
            assert 'requires a nonempty key' in json.dumps(data['messages'])
        names = {t['function']['name'] for t in data['tools']}
        assert {'read_file', 'edit_file', 'write_file', 'bash'} <= names
        return 200, response([actions[number - 1]])
    with local_provider(reply) as (url, requests):
        agent = make_agent(tmp_path, monkeypatch, url, extensions=['scratchpad', 'trajectory_read'],
                           budget=Budget(.1, 300000, 16), task_intent='fix', step_limit=16)
        assert agent.run('Fix addition including negative operands.')['exit_status'] == 'Submitted'
    assert len(requests) == 12 and agent.env.actions == 11
    receipts = [read_json(p) for p in sorted((tmp_path / 'agent/actions').glob('*.json'))]
    assert sum(bool(r.get('workspace_changed')) for r in receipts) == 2
    assert sum(bool(r.get('verification_candidate')) for r in receipts) == 2
    assert not agent.env.progress.data['unresolved_commands']
    export_artifact(tmp_path, {'reason': 'public scripted control-chain development fixture'})
    assert verify_artifact(tmp_path)['autonomous_complete']
    assert (tmp_path / 'model.patch').stat().st_size > 0
