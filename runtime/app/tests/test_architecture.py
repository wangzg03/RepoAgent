"""Request architecture over public development fixtures, never benchmark answers."""

import copy
import json
from types import SimpleNamespace

import pytest
from test_loop import create_case
from test_native_upstream import local_provider, make_agent, response

from corecoder_mini.budget import Budget, ChildBudget
from corecoder_mini.context_view import ContextView, message_groups
from corecoder_mini.continuation import ActionContinuation
from corecoder_mini.delivery import export_artifact, read_json
from corecoder_mini.engine import SUBMIT
from corecoder_mini.events import Events
from corecoder_mini.progress import WorkspaceProgress
from corecoder_mini.runtime import local
from corecoder_mini.state_tools import WorkingState
from corecoder_mini.supervisor import verify_artifact
from corecoder_mini.token_policy import RuntimeGuidance


@pytest.fixture(autouse=True)
def clean_runtime():
    vars(local).clear()
    yield
    vars(local).clear()


@pytest.mark.parametrize('cap,spent,stage', [(6, 2, 'economy'), (6, 3, 'strict'), (6, 5, 'finalize'),
                                          (24, 8, 'economy'), (24, 12, 'strict'), (24, 20, 'finalize')])
def test_shared_request_budget_reaches_convergence_before_tool_limit(cap, spent, stage):
    budget = Budget(1, 500000, cap)
    budget.requests = spent
    before = {k: v for k, v in vars(budget).items() if isinstance(v, (int, float)) or v is None}
    feedback = RuntimeGuidance(SimpleNamespace(actions=2), SimpleNamespace(budget=budget),
                               {'tool_limit': 48, 'agent_seconds': 3600})()
    assert feedback['stage'] == stage and feedback['budget']['requests_remaining'] == cap - spent
    assert before == {k: v for k, v in vars(budget).items() if isinstance(v, (int, float)) or v is None}


def test_token_pressure_not_tool_count_drives_same_stage():
    budget = Budget(1, 10000, 100)
    budget.tokens_spent = 8100
    feedback = RuntimeGuidance(SimpleNamespace(actions=1), SimpleNamespace(budget=budget),
                               {'tool_limit': 48, 'agent_seconds': 3600})()
    assert feedback['stage'] == 'finalize' and feedback['budget']['tokens_remaining'] == 1900


def test_native_child_sees_its_remaining_delivery_request():
    shared = Budget(1, 500000, 24)
    child = ChildBudget(shared, request_limit=3, reserve_delivery=True)
    child.requests = 2
    feedback = RuntimeGuidance(SimpleNamespace(actions=1), SimpleNamespace(budget=child),
                               {'tool_limit': 48, 'agent_seconds': 120})()
    assert feedback['stage'] == 'strict' and feedback['budget']['child_requests_remaining'] == 1
    assert shared.requests == 0 and child.requests == 2 and not child.delivering


def setup_controller(tmp_path, kind='fix'):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    (workspace / 'calc.py').write_text('def add(a,b):\n    return a-b\n')
    state = WorkingState(tmp_path / 'agent')
    state.bind_task_contract(kind)
    progress = WorkspaceProgress(workspace, tmp_path / 'agent', state=state)
    progress.observe(1, 'read_file', {'file_path': 'calc.py'}, {'returncode': 0, 'output': 'PUBLIC_SOURCE'})
    registry = SimpleNamespace(tools={'bash': None, 'edit_file': None, 'read_file': None, 'trajectory_read': None})
    return state, progress, registry


def test_recovery_changes_advice_without_restricting_tools_or_faking_edits(tmp_path):
    state, progress, registry = setup_controller(tmp_path)
    progress.data['recovery']['active_window'] = {'phase': 'converge'}
    before = copy.deepcopy(progress.data)
    original_tools = list(registry.tools)
    packet = ActionContinuation(progress, registry)()
    assert 'smallest supported change now' in packet['instruction']
    assert progress.data == before and list(registry.tools) == original_tools
    assert not packet['facts']['source_changes'] and not state.data['todos']
    assert not (progress.directory / 'submission.json').exists()


@pytest.mark.parametrize('kind', ['explain', 'review', 'locate'])
def test_budget_pressure_never_turns_read_only_work_into_editing(tmp_path, kind):
    _state, progress, registry = setup_controller(tmp_path, kind)
    packet = ActionContinuation(progress, registry, lambda: {
        'stage': 'finalize', 'budget': {'requests_remaining': 1}})()
    assert packet['status'] == 'deliver_findings' and 'do not edit source' in packet['instruction']
    assert 'make the smallest' not in packet['instruction']


def test_request_projection_drops_old_recovery_dump_but_keeps_live_memory_and_tool_text(tmp_path):
    state, progress, registry = setup_controller(tmp_path)
    state.update(lambda d: d['notes'].update(hypothesis='PUBLIC_HYPOTHESIS', next_action='PUBLIC_NEXT_ACTION'))
    packet = {'phase': 'converge', 'reanchor_id': 1, 'idle_requests': 14, 'remaining_repair_requests': 0,
              'recovery_packet': {'actions': [{'sha256': '0' * 64, 'stale_record': 'OLD_RECOVERY'}] * 10},
              'required_next_step': 'DUPLICATE_INSTRUCTION'}
    view = ContextView(progress.directory, state, Events(progress.directory), progress=progress.feedback,
                       recovery=lambda: packet)
    view.continuation = ActionContinuation(progress, registry)
    messages = [{'role': 'system', 'content': 'PUBLIC_POLICY'}, {'role': 'user', 'content': 'PUBLIC_TASK'},
                {'role': 'assistant', 'content': 'PUBLIC_ACTION_SUMMARY', 'tool_calls': [
                    {'id': 'one', 'type': 'function', 'function': {'name': 'read_file',
                     'arguments': '{"file_path":"calc.py"}'}}]},
                {'role': 'tool', 'tool_call_id': 'one', 'content': 'EXACT_CURRENT_SOURCE',
                 'extra': {'action_id': 1, 'returncode': 0, 'observation_kind': 'inspection'}}]
    original = copy.deepcopy(messages)
    wire, receipt = view.project(messages, 8000)
    text = json.dumps(wire)
    assert 'OLD_RECOVERY' not in text and 'DUPLICATE_INSTRUCTION' not in text
    assert all(s in text for s in ['PUBLIC_HYPOTHESIS', 'PUBLIC_NEXT_ACTION', 'EXACT_CURRENT_SOURCE', 'PUBLIC_TASK'])
    assert messages == original == read_json(progress.directory / 'canonical-messages.json')
    assert len(message_groups(wire)[1]) == 1 and not receipt['compacted']
    assert len(wire[-2]['content']) < len(json.dumps(packet))


@pytest.mark.docker
def test_actual_native_loop_converges_then_edits_tests_repairs_submits(tmp_path, monkeypatch):
    create_case(tmp_path)
    check = "python -c 'from calc import add; assert add(2,3)==5; assert add(-1,1)==0'"
    calls = [('read_file', {'file_path': 'calc.py'}),
             ('bash', {'command': 'wc -l calc.py'}),
             ('bash', {'command': 'head -n 2 calc.py'}),
             ('edit_file', {'file_path': 'calc.py', 'old_string': 'return a - b', 'new_string': 'return a * b'}),
             ('bash', {'command': check}),
             ('edit_file', {'file_path': 'calc.py', 'old_string': 'return a * b', 'new_string': 'return a + b'}),
             ('bash', {'command': check}), ('bash', {'command': SUBMIT})]
    seen_tools = []
    def reply(data, number):
        seen_tools.append([t['function']['name'] for t in data['tools']])
        assert 'concise action summary' in data['messages'][0]['content']
        if number == 4:
            assert 'smallest supported change now' in data['messages'][-1]['content']
        if number == 6:
            assert 'Repair the recorded failed test/assertion' in data['messages'][-1]['content']
        assert 'recovery_packet' not in json.dumps(data['messages'])
        return 200, response([calls[number - 1]])
    with local_provider(reply) as (url, requests):
        agent = make_agent(tmp_path, monkeypatch, url, extensions=['scratchpad', 'todo_write', 'trajectory_read'],
                           budget=Budget(1, 500000, 10), task_intent='fix', request_limit=10,
                           stall_request_limit=2, repair_window_requests=1, reasoning_effort='none')
        try:
            result = agent.run('Fix add so it returns the sum; test positive and opposite-sign values.')
        finally:
            agent.env.cleanup()
        assert result['exit_status'] == 'Submitted' and len(requests) == 8
        assert all(names == seen_tools[0] for names in seen_tools)
        export_artifact(tmp_path, {'state': 'submitted'})
        acceptance = verify_artifact(tmp_path)
        assert acceptance['artifact_correct'] is True
        assert read_json(tmp_path / 'agent/submission.json')['model_submitted']
