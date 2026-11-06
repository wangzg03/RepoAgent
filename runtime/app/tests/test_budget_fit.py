"""Public request fixtures: no benchmark traces, answers or paid providers."""

import copy
import json

import pytest
from test_harness import history
from test_loop import create_case
from test_native_upstream import local_provider, make_agent, response

from corecoder_mini.budget import Budget, BudgetExceeded
from corecoder_mini.context_view import ContextView, TrajectoryTool, message_groups
from corecoder_mini.delivery import export_artifact, read_json
from corecoder_mini.engine import SUBMIT
from corecoder_mini.events import Events
from corecoder_mini.runtime import local
from corecoder_mini.state_tools import ScratchpadTool, WorkingState
from corecoder_mini.supervisor import verify_artifact


@pytest.fixture(autouse=True)
def runtime():
    vars(local).clear()
    yield
    vars(local).clear()


def fixture_history():
    messages = history(10)
    for index in range(10):
        messages[3 + index * 2]['content'] = f'PUBLIC_SOURCE_{index}\n' + 'unique observation ' * 220
        messages[3 + index * 2]['extra']['returncode'] = 0
    messages[-5]['extra']['workspace_changed'] = True
    messages[-3]['extra'].update(verification_candidate=True, returncode=1)
    messages[-3]['content'] = 'AssertionError: add(2, 3) produced 6; expected 5'
    return messages


def make_view(path, **kwargs):
    state = WorkingState(path)
    ScratchpadTool(state).execute('write', 'hypothesis', 'Repair arithmetic, then rerun positive and signed cases')
    return ContextView(path, state, Events(path), target_tokens=100000,
                       history_output_chars=0, **kwargs)


def preview(budget, rates=(1, 2)):
    return lambda rows: budget.admission({'messages': rows, 'max_tokens': 1024}, rates)


@pytest.mark.parametrize('kind', ['tokens', 'usd'])
def test_window_fits_but_budget_requires_archiving_with_exact_repair_context(tmp_path, kind):
    messages = fixture_history()
    original = copy.deepcopy(messages)
    view = make_view(tmp_path)
    full, _ = view.project(messages, 100000)
    quote = Budget.quote({'messages': full, 'max_tokens': 1024}, (1, 2))
    budget = Budget(1 if kind == 'tokens' else float(quote) * .75,
                    int(quote.tokens * .75) if kind == 'tokens' else None)
    wire, receipt = view.project(messages, 100000, admission=preview(budget))
    fit = receipt['budget_fit']
    assert not fit['initial']['allowed'] and fit['final']['allowed']
    assert fit['archived_turns'] and not receipt['checkpoint_reused']
    assert budget.requests == budget.tokens_spent == 0
    assert messages == original == read_json(tmp_path/'canonical-messages.json')
    groups = message_groups(wire)[1]
    assert [g[0]['tool_calls'][0]['id'] for g in groups[-3:]] == ['8', '9', '10']
    assert groups[-3][1]['content'] == messages[-5]['content']
    assert groups[-2][1]['content'] == messages[-3]['content']
    assert groups[-1][1]['content'] == messages[-1]['content']
    assert wire[:2] == messages[:2] and 'Repair arithmetic' in wire[-1]['content']
    recalled = json.loads(TrajectoryTool(tmp_path).execute(turn=fit['archived_turns'][0], limit=4000))
    assert recalled[0]['observations'][0]['content'].startswith('PUBLIC_SOURCE_')
    assert all(group[0]['reasoning_content'] == 'opaque' for group in groups)


def test_reused_checkpoint_is_rechecked_after_shared_budget_consumption(tmp_path):
    messages = fixture_history()
    view = make_view(tmp_path)
    full, _ = view.project(messages, 100000)
    cost = Budget.quote({'messages': full, 'max_tokens': 1024}, (1, 2))
    budget = Budget(1, cost.tokens + 20000)
    _, first = view.project(messages, 100000, admission=preview(budget))
    assert first['checkpoint_reused'] and not first['budget_fit']['archived_turns']
    # A concurrent participant completed work against the same ledger.
    r = budget.reserve({'messages': [{'role': 'user', 'content': 'x' * 30000}], 'max_tokens': 1024}, (1, 2))
    budget.settle(r, {'prompt_tokens': 29000, 'completion_tokens': 100}, (1, 2))
    restored = make_view(tmp_path)
    _, second = restored.project(messages, 100000, admission=preview(budget))
    assert second['budget_fit']['archived_turns'] and second['budget_fit']['final']['allowed']
    assert not second['checkpoint_reused']


@pytest.mark.parametrize('blocker', ['requests', 'provider_bound_violation'])
def test_non_payload_blocker_does_not_discard_history(tmp_path, blocker):
    view = make_view(tmp_path)
    messages = fixture_history()
    budget = Budget(1, request_limit=1)
    if blocker == 'requests':
        r = budget.reserve({'messages': [], 'max_tokens': 1024}, (1, 2))
        budget.settle(r, None, (1, 2))
    else:
        budget.blocked = True
    wire, receipt = view.project(messages, 100000, admission=preview(budget))
    assert not receipt['budget_fit']['archived_turns']
    assert receipt['budget_fit']['final']['blocking_limits'] == [blocker]
    assert len(message_groups(wire)[1]) == 10


def test_no_recall_does_not_drop_evidence_to_fit_budget(tmp_path):
    messages = fixture_history()
    view = make_view(tmp_path, allow_recall=False)
    wire, receipt = view.project(messages, 100000, admission=preview(Budget(1, 10000)))
    assert len(message_groups(wire)[1]) == 10 and not receipt['compacted']
    assert not preview(Budget(1, 10000))(wire)['allowed']


def test_native_http_receives_fitted_context_and_unchanged_output_cap(tmp_path, monkeypatch):
    budget = Budget(1, 38000, 3)
    def reply(data, number):
        assert data['max_tokens'] == 8192
        assert 'AssertionError: add(2, 3)' in json.dumps(data['messages'])
        assert 'Repair arithmetic' in json.dumps(data['messages'])
        return 200, response([('edit_file', {'file_path': 'calc.py', 'old_string': 'return a * b', 'new_string': 'return a + b'})])
    with local_provider(reply) as (url, requests):
        agent = make_agent(tmp_path, monkeypatch, url, extensions=['trajectory_read'], budget=budget,
                           max_output_tokens=8192, context_tokens=256000)
        agent.model.view = make_view(tmp_path/'agent')
        result = agent.model.query(fixture_history())
        receipt = read_json(tmp_path/'agent/requests/0001.json')
        assert len(requests) == budget.requests == 1
        assert receipt['projection']['budget_fit']['archived_turns']
        assert receipt['status'] == 'returned'
        assert result['extra']['actions'][0]['tool'] == 'edit_file'
        assert budget.token_limit == 38000


def test_mandatory_context_too_large_still_blocks_before_http(tmp_path, monkeypatch):
    with local_provider(lambda *a: pytest.fail('No HTTP allowed')) as (url, requests):
        agent = make_agent(tmp_path, monkeypatch, url, extensions=['trajectory_read'],
                           budget=Budget(1, 1000, 3), context_tokens=256000)
        agent.model.view = make_view(tmp_path/'agent')
        with pytest.raises(BudgetExceeded):
            agent.model.query(fixture_history())
        receipt = read_json(tmp_path/'agent/requests/0001.json')
        assert not requests and receipt['status'] == 'blocked_before_transport'
        assert not receipt['projection']['budget_fit']['final']['allowed']
        wire = read_json(tmp_path/'agent/private/request-views/0001.json')['messages']
        assert len(message_groups(wire)[1]) == 3
        assert 'AssertionError: add(2, 3)' in json.dumps(wire)


@pytest.mark.docker
def test_budget_projection_allows_real_edit_failed_check_repair_and_delivery(tmp_path, monkeypatch):
    create_case(tmp_path)
    check = "python -c 'from calc import add; assert add(2,3)==5; assert add(-1,1)==0'"
    calls = [('bash', {'command': 'python -c "print(\'public observation a \' * 400)"'}),
             ('bash', {'command': 'python -c "print(\'public observation b \' * 400)"'}),
             ('edit_file', {'file_path': 'calc.py', 'old_string': 'return a - b', 'new_string': 'return a * b'}),
             ('bash', {'command': check}),
             ('edit_file', {'file_path': 'calc.py', 'old_string': 'return a * b', 'new_string': 'return a + b'}),
             ('bash', {'command': check}), ('bash', {'command': SUBMIT})]
    budget = Budget(1, 40000, 9)
    def reply(data, number):
        if number == 5:
            assert 'AssertionError' in json.dumps(data['messages'])
        payload = response([calls[number - 1]])
        payload['usage'] = {'prompt_tokens': 4000, 'completion_tokens': 100, 'total_tokens': 4100}
        return 200, payload
    with local_provider(reply) as (url, requests):
        agent = make_agent(tmp_path, monkeypatch, url, extensions=['trajectory_read'], budget=budget,
                           context_tokens=256000, max_output_tokens=1024, history_target_tokens=200000,
                           history_output_chars=0, task_intent='fix', request_limit=9)
        try:
            result = agent.run('Fix add to return the sum. Check positive and opposite-sign cases.')
        finally:
            agent.env.cleanup()
        assert result['exit_status'] == 'Submitted' and len(requests) == 7
        receipts = [read_json(p) for p in (tmp_path/'agent/requests').glob('*.json')]
        assert any(r['projection']['budget_fit']['archived_turns'] for r in receipts)
        assert budget.token_limit == 40000 and budget.tokens_spent == 28700
        export_artifact(tmp_path, {'state': 'submitted'})
        assert verify_artifact(tmp_path)['artifact_correct']
