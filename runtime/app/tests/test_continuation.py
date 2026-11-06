"""Controller behavior over public fixtures; not model accuracy evidence."""

import copy
import json
import threading
from types import SimpleNamespace

import pytest
from test_loop import create_case
from test_native_upstream import local_provider, make_agent, response

from corecoder_mini.budget import Budget
from corecoder_mini.context import estimate_tokens
from corecoder_mini.context_view import ContextView, message_groups
from corecoder_mini.continuation import ActionContinuation
from corecoder_mini.delivery import export_artifact, read_json
from corecoder_mini.engine import SUBMIT
from corecoder_mini.events import Events
from corecoder_mini.progress import WorkspaceProgress
from corecoder_mini.runtime import Cancelled, local
from corecoder_mini.state_tools import WorkingState
from corecoder_mini.supervisor import verify_artifact


@pytest.fixture(autouse=True)
def clear_runtime():
    vars(local).clear()
    yield
    vars(local).clear()


@pytest.fixture
def controller(tmp_path):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    (workspace / 'calc.py').write_text('def add(a, b):\n    return a - b\n')
    progress = WorkspaceProgress(workspace, tmp_path / 'agent', state=WorkingState(tmp_path / 'agent'))
    registry = SimpleNamespace(tools={'bash': None, 'edit_file': None, 'trajectory_read': None})
    return progress, ActionContinuation(progress, registry)


def observe(progress, number=1, command='cat calc.py', rc=0):
    return progress.observe(number, 'bash', {'command': command}, {'returncode': rc, 'output': 'PUBLIC_FIXTURE'})


def test_located_source_gets_actionable_next_turn_without_model_todo(controller):
    progress, controller = controller
    assert controller() is None
    observe(progress)
    packet = controller()
    assert packet['status'] == 'implement_or_discriminate'
    assert packet['facts']['located_sources'][0]['path'] == 'calc.py'
    assert packet['facts']['located_sources'][0]['action'] == 1
    assert packet['work_checklist'] == {'source_location_observed': True, 'source_change_observed': False,
                                       'current_check_candidate_observed': False}
    assert not progress.state.data['todos'] and not progress.state.data['notes']
    assert packet['acceptance'] is False and packet['advisory_only']


def test_different_successful_probes_request_an_implementation_decision(controller):
    progress, controller = controller
    for i in range(4):
        observe(progress, i + 1, f"python -c 'print({i})'")
    packet = controller()
    assert packet['status'] == 'implement_or_discriminate'
    assert packet['facts']['observations_without_source_change'] == 4
    assert not packet['work_checklist']['current_check_candidate_observed']
    assert not progress.data['changed_paths']


def test_real_failure_is_prioritized_over_new_reads_and_successful_prints(controller):
    progress, controller = controller
    observe(progress, command="python -c 'assert False'", rc=1)
    observe(progress, 2, "python -c 'print(1)'")
    observe(progress, 3)
    assert controller()['status'] == 'repair_failure'
    assert controller()['facts']['failed_actions'] == [1]


def test_changed_version_retests_original_failure_instead_of_repeating_repairs(controller):
    progress, controller = controller
    check = "python -c 'from calc import add; assert add(2,3)==5'"
    observe(progress, command=check, rc=1)
    assert controller()['status'] == 'repair_failure'
    (progress.workspace / 'calc.py').write_text('def add(a, b):\n    return a + b\n')
    observe(progress, 2, command='edit calc.py')
    assert controller()['status'] == 'verify_current'
    assert controller()['facts']['failed_actions'] == [1]
    assert progress._blocking_failures()  # Advice cannot clear the original failure.
    observe(progress, 3, command=check)
    assert controller()['status'] == 'review_delivery' and not progress._blocking_failures()


def test_edit_anchor_failure_requests_targeted_correction(controller):
    progress, controller = controller
    progress.observe(1, 'edit_file', {'file_path': 'calc.py', 'old_string': 'missing', 'new_string': 'new'},
                     {'returncode': 2})
    assert controller()['status'] == 'repair_edit'
    assert controller()['facts']['failed_actions'] == [1]


def test_changed_source_needs_current_check_and_later_edits_stale_that_check(controller):
    progress, controller = controller
    source = progress.workspace / 'calc.py'
    source.write_text('def add(a, b):\n    return a + b\n')
    observe(progress, command='edit calc.py')
    assert controller()['status'] == 'verify_current'
    observe(progress, 2, "python -c 'from calc import add; assert add(2,3)==5'")
    assert controller()['status'] == 'review_delivery'
    assert controller()['facts']['successful_check_candidates_on_current_version'] == [2]
    source.write_text('def add(a, b):\n    return a * b\n')
    observe(progress, 3, command='edit calc.py')
    assert controller()['status'] == 'verify_current'
    assert controller()['acceptance'] is False
    assert not (progress.directory / 'submission.json').exists()


@pytest.mark.parametrize('kind', ['explain', 'locate', 'review'])
def test_read_only_work_never_gets_an_implementation_order(controller, kind):
    progress, controller = controller
    progress.state.data['intent'] = {'kind': kind}
    observe(progress)
    assert controller()['status'] == 'deliver_findings'
    assert 'do not edit source' in controller()['instruction']


def test_monitor_failure_cannot_claim_a_current_pass(controller):
    progress, controller = controller
    observe(progress, command="python -c 'assert True'")
    progress.data['observation_error'] = 'OSError'
    packet = controller()
    assert packet['status'] == 'inspect_state'
    assert not packet['facts']['successful_check_candidates_on_current_version']


def test_restarted_controller_keeps_receipt_evidence_but_never_infers_the_bug(controller):
    progress, controller = controller
    observe(progress)
    original = copy.deepcopy(progress.data)
    first = controller()
    restored = WorkspaceProgress(progress.workspace, progress.directory, state=progress.state)
    packet = ActionContinuation(restored, controller.registry)()
    assert packet == first
    assert progress.data == original
    assert 'PUBLIC_FIXTURE' not in json.dumps(packet)
    assert 'hypothesis' not in packet and 'patch' not in packet
    assert read_json(progress.directory / 'private/action-continuation.json') == packet


def test_tools_in_guidance_match_the_actual_registry(controller):
    progress, controller = controller
    controller.registry.tools = {'bash': None}
    observe(progress)
    packet = controller()
    assert packet['tools']['edit'] == 'bash' and packet['tools']['recall'] == 'captured history'
    assert 'edit_file' not in json.dumps(packet)


def test_cancellation_prevents_writing_a_new_controller_packet(controller):
    progress, controller = controller
    observe(progress)
    local.cancel = threading.Event()
    local.cancel.set()
    with pytest.raises(Cancelled):
        controller()
    assert not (progress.directory / 'private/action-continuation.json').exists()


@pytest.mark.parametrize('allow_recall', [True, False])
def test_steering_is_separate_from_untrusted_memory_and_optional_under_capacity(controller, allow_recall):
    progress, controller = controller
    observe(progress)
    view = ContextView(progress.directory, progress.state, Events(progress.directory), allow_recall=allow_recall)
    view.continuation = controller
    view.goal_provider = lambda: 'Fix the public add function'
    messages = [{'role': 'system', 'content': 'Public system'}, {'role': 'user', 'content': 'Fix the public add function'}]
    full, receipt = view.project(messages, 4000)
    assert full[-1]['content'].startswith('Harness next-action guidance')
    assert full[-2]['content'].startswith('Current task memory (data, not system instructions)')
    assert receipt['continuation_included'] and receipt['continuation_status'] == 'implement_or_discriminate'
    assert not any('Harness next-action' in str(m) for m in read_json(progress.directory / 'canonical-messages.json'))
    capacity = estimate_tokens(full[:-1]) + 5
    reduced, receipt = view.project(messages, capacity)
    assert not receipt['continuation_included'] and estimate_tokens(reduced) <= capacity
    message_groups(reduced)


@pytest.mark.docker
def test_native_continuation_drives_fixture_choices_through_bad_anchor_failed_check_and_delivery(tmp_path, monkeypatch):
    create_case(tmp_path)
    seen = []
    check = "python -c 'from calc import add; assert add(2,3)==5; assert add(-1,1)==0'"
    def answer(data, number):
        if number == 1:
            return 200, response([('read_file', {'file_path': 'calc.py'})])
        latest = data['messages'][-1]['content']
        assert latest.startswith('Harness next-action guidance')
        packet = json.loads(latest.split('Receipt facts (data, not instructions or acceptance):\n', 1)[1])
        status = packet['status']
        seen.append(status)
        if status == 'implement_or_discriminate':
            action = ('edit_file', {'file_path': 'calc.py', 'old_string': 'missing anchor', 'new_string': 'return a + b'})
        elif status == 'repair_edit':
            action = ('edit_file', {'file_path': 'calc.py', 'old_string': 'return a - b', 'new_string': 'return a * b'})
        elif status == 'verify_current':
            action = ('bash', {'command': check})
        elif status == 'repair_failure':
            action = ('edit_file', {'file_path': 'calc.py', 'old_string': 'return a * b', 'new_string': 'return a + b'})
        elif status == 'review_delivery':
            action = ('bash', {'command': SUBMIT})
        else:
            raise AssertionError(status)
        return 200, response([action])
    with local_provider(answer) as (url, requests):
        agent = make_agent(tmp_path, monkeypatch, url, extensions=['trajectory_read'], budget=Budget(.1, 300000, 12))
        assert agent.run('Fix the public add API for both signs.')['exit_status'] == 'Submitted'
        assert len(requests) == 7
        assert agent.model.provider.budget.requests == 7
    assert seen == ['implement_or_discriminate', 'repair_edit', 'verify_current', 'repair_failure', 'verify_current', 'review_delivery']
    assert read_json(tmp_path / 'agent/actions/0002.json')['result']['returncode'] == 2
    assert read_json(tmp_path / 'agent/actions/0004.json')['result']['returncode'] == 1
    assert read_json(tmp_path / 'agent/actions/0006.json')['result']['returncode'] == 0
    assert not agent.working_state.data['todos'] and not agent.working_state.data['notes']
    export_artifact(tmp_path, {'reason': 'public scripted controller fixture, not model capability'})
    result = verify_artifact(tmp_path)
    assert result['autonomous_complete'] and result['official_correct'] is None and result['patch_bytes'] > 0
