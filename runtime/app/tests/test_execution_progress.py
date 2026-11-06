"""Independent development fixtures; not model accuracy or SWE-bench scores."""

import copy
import json

import pytest
from test_harness import history
from test_loop import create_case
from test_native_upstream import local_provider, make_agent, response

from corecoder_mini.budget import Budget
from corecoder_mini.context_view import ContextView, message_groups
from corecoder_mini.delivery import export_artifact, read_json
from corecoder_mini.engine import SUBMIT
from corecoder_mini.events import Events
from corecoder_mini.progress import ProgressStalled, WorkspaceProgress
from corecoder_mini.runtime import local
from corecoder_mini.state_tools import WorkingState
from corecoder_mini.supervisor import verify_artifact


@pytest.fixture(autouse=True)
def clear_runtime():
    vars(local).clear()
    yield
    vars(local).clear()


@pytest.mark.parametrize('program', [
    "python -c 'print({n})'",
    "cd /workspace && python - <<'PY'\nprint({n})\nPY",
    "cd /workspace && env X=1 python -c 'print({n})'",
    'custom_diagnostic --case {n}',
])
def test_distinct_executed_diagnostics_survive_recovery_without_becoming_checks(tmp_path, program):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    settings = {'stall_request_limit': 2, 'repair_window_requests': 2}
    for n in range(12):
        # Recreate the monitor between observations: novelty must survive restart.
        progress = WorkspaceProgress(workspace, tmp_path / 'agent', settings)
        progress.request_feedback()
        annotation = progress.observe(n + 1, 'bash', {'command': program.format(n=n)},
                                      {'returncode': 0, 'output': str(n)})
        assert annotation['new_observation_evidence']
        assert not annotation['verification_candidate'] and not annotation['workspace_changed']
    feedback = progress.request_feedback()
    assert feedback['phase'] == 'converge' and feedback['no_information_requests'] == 0
    assert 'Implementation is still missing' in feedback['next_action']
    assert progress.data['productive_epoch'] == 0 and progress.data['evidence_epoch'] == 12
    assert len(progress.data['recovery']['reanchors']) == 1


@pytest.mark.parametrize('commands', [
    ["python -c 'print(1)'", "cd /workspace && python -c 'print(1)'", "python -c 'print( 1 )' # another thought"],
    ['custom_diagnostic --case 1', 'custom_diagnostic --case 2'],
])
def test_repeated_diagnostics_and_cycles_stop_despite_changing_stdout(tmp_path, commands):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    progress = WorkspaceProgress(workspace, tmp_path / 'agent', {'stall_request_limit': 2, 'repair_window_requests': 2})
    for i in range(12):
        try:
            progress.request_feedback()
        except ProgressStalled:
            break
        progress.observe(i + 1, 'bash', {'command': commands[i % len(commands)]},
                         {'returncode': 0, 'output': f'elapsed={i}'})
    else:
        pytest.fail('Repeated observations must not keep the agent alive until the hard request cap')
    assert progress.data['recovery']['stop_cause'] == 'repeated_observations_or_bookkeeping'
    assert not progress.data['changed_paths'] and not progress.data['check_signatures']


def test_distinct_diagnostics_do_not_waive_an_actual_failed_check_window(tmp_path):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    progress = WorkspaceProgress(workspace, tmp_path / 'agent', {'stall_request_limit': 2, 'repair_window_requests': 2})
    progress.observe(1, 'bash', {'command': "python -c 'assert False'"},
                     {'returncode': 1, 'output': 'AssertionError'})
    for n in range(10):
        try:
            feedback = progress.request_feedback()
        except ProgressStalled:
            break
        if feedback:
            assert 'recorded failed' in feedback['next_action']
        progress.observe(n + 2, 'bash', {'command': f"python -c 'print({n})'"},
                         {'returncode': 0, 'output': str(n)})
    else:
        pytest.fail('New reads must not erase the unchanged failed check')
    assert progress.data['recovery']['stop_cause'] == 'unresolved_execution_failure'
    assert progress.data['unresolved_commands'][0]['action_id'] == 1
    assert progress.data['recovery']['no_information_requests'] == 0


def test_delivery_advice_follows_current_source_and_check_receipts(tmp_path):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    source = workspace / 'calc.py'
    source.write_text('value = 1\n')
    progress = WorkspaceProgress(workspace, tmp_path / 'agent', {'stall_request_limit': 1, 'repair_window_requests': 3})
    assert 'Implementation is still missing' in progress.request_feedback()['next_action']
    source.write_text('value = 2\n')
    progress.observe(1, 'bash', {'command': 'edit calc.py'}, {'returncode': 0})
    assert 'Source is changed' in progress.request_feedback()['next_action']
    progress.observe(2, 'bash', {'command': "python -c 'assert True'"}, {'returncode': 0})
    assert 'submit with echo' in progress.request_feedback()['next_action']
    source.write_text('value = 3\n')
    progress.observe(3, 'bash', {'command': 'edit calc.py'}, {'returncode': 0})
    assert 'Source is changed' in progress.request_feedback()['next_action']
    assert not (tmp_path / 'agent/submission.json').exists()  # Advice never submits.


def test_read_only_intent_never_receives_an_edit_instruction_from_status(tmp_path):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    state = WorkingState(tmp_path / 'agent')
    state.data['intent'] = {'kind': 'explain'}
    progress = WorkspaceProgress(workspace, tmp_path / 'agent', {'stall_request_limit': 1}, state)
    feedback = progress.request_feedback()
    assert feedback['read_only_intent'] and 'do not change source' in feedback['next_action']


@pytest.mark.parametrize('recover', [False, True])
def test_projection_under_pressure_keeps_early_python_diagnosis_and_recent_experiments(tmp_path, recover):
    messages = history(18)
    for index in range(18):
        tool = messages[3 + index * 2]
        tool['extra'].update(action_id=index + 1, observation_kind='inspection')
    for index in [0, 1, 10, 13]:
        tool = messages[3 + index * 2]
        tool['extra']['observation_kind'] = 'execution'
        tool['content'] = f'ACTUAL_PROBE_{index}: observed behavior differs from expected'
    original = copy.deepcopy(messages)
    view = ContextView(tmp_path, WorkingState(tmp_path), Events(tmp_path), target_tokens=3000,
                       history_output_chars=1000, recovery=(lambda: {'reanchor_id': 1}) if recover else None)
    projected, receipt = view.project(messages, 14000)
    assert receipt['compacted'] and {1, 2, 11, 14, 15, 16, 17, 18} <= set(receipt['retained_turns'])
    for index in [0, 1, 10, 13]:
        assert f'ACTUAL_PROBE_{index}' in json.dumps(projected)
    message_groups(projected)
    assert messages == original == read_json(tmp_path / 'canonical-messages.json')
    if recover:
        assert read_json(tmp_path / 'private/action-recovery/0001.json')['history_pruned']
    # A later request must keep the diagnosis, not undo recovery by normal compaction.
    later, _ = view.project(messages + history(1)[2:], 14000)
    assert 'ACTUAL_PROBE_0' in json.dumps(later) and 'ACTUAL_PROBE_13' in json.dumps(later)


def test_diagnostic_anchors_keep_exact_output_and_fail_capacity_instead_of_silent_loss(tmp_path):
    messages = history(10)
    diagnosis = 'FIRST_DIAGNOSTIC\n' + 'important detail ' * 500 + '\nEND_DIAGNOSTIC'
    messages[3]['content'] = diagnosis
    messages[3]['extra'].update(observation_kind='execution')
    view = ContextView(tmp_path, WorkingState(tmp_path), Events(tmp_path), target_tokens=2000,
                       history_output_chars=1000)
    projected, receipt = view.project(messages, 20000)
    assert diagnosis in [m.get('content') for m in projected]
    assert not any(r['turn'] == 1 for r in receipt['output_reductions'])
    with pytest.raises(RuntimeError, match='context_capacity'):
        view.project(messages, 500)


@pytest.mark.docker
def test_native_python_investigation_can_edit_repair_verify_and_submit_after_old_cutoff(tmp_path, monkeypatch):
    create_case(tmp_path)
    check = "python -c 'from calc import add; assert add(2,3)==5; assert add(-1,1)==0'"
    steps = [('bash', {'command': f"python -c 'from calc import add; print((\"probe-{n}\", add({n},1)))'"})
             for n in range(14)]
    steps += [('bash', {'command': "sed -i 's/a - b/a * b/' calc.py"}),
              ('bash', {'command': check}),
              ('bash', {'command': r"sed -i 's/a \* b/a + b/' calc.py"}),
              ('bash', {'command': check}), ('bash', {'command': SUBMIT})]

    def answer(data, number):
        if number == 15:
            wire = json.dumps(data['messages'])
            assert 'probe-0' in wire and 'probe-13' in wire
            assert 'Implementation is still missing' in wire and 'recovery_packet' not in wire
            assert read_json(tmp_path / 'agent/workspace-progress.json')['recovery']['reanchors']
            message_groups(data['messages'])
        if number == 17:
            assert 'AssertionError' in json.dumps(data['messages'])
        return 200, response([steps[number - 1]])

    with local_provider(answer) as (url, requests):
        agent = make_agent(tmp_path, monkeypatch, url, extensions=['scratchpad', 'todo_write', 'trajectory_read'],
                           context_tokens=32000, budget=Budget(.1, 300000, 24))
        assert agent.run('Fix the public add API and verify both signs.')['exit_status'] == 'Submitted'
        assert len(requests) == agent.model.provider.budget.requests == 19
    assert all(not read_json(tmp_path / f'agent/actions/{n:04d}.json')['verification_candidate'] for n in range(1, 15))
    assert read_json(tmp_path / 'agent/actions/0016.json')['result']['returncode'] == 1
    assert read_json(tmp_path / 'agent/actions/0018.json')['result']['returncode'] == 0
    assert not agent.env.progress.data['unresolved_commands']
    assert agent.env.progress.data['changed_paths'] == ['calc.py']
    recovery = read_json(tmp_path / 'agent/private/action-recovery/0001.json')
    assert not recovery['history_pruned'] and recovery['replayed_tools'] == 0
    export_artifact(tmp_path, {'reason': 'public scripted protocol fixture, not a model score'})
    result = verify_artifact(tmp_path)
    assert result['model_submitted'] and result['artifact_correct'] and result['autonomous_complete']
    assert result['official_correct'] is None and result['patch_bytes'] > 0
