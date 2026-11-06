"""Bounded retention of exact evidence with pageable, protocol-valid archives."""

import copy
import json

import pytest
from test_harness import history
from test_loop import create_case
from test_native_upstream import local_provider, make_agent, response

from corecoder_mini.budget import Budget
from corecoder_mini.context import estimate_tokens
from corecoder_mini.context_view import ContextView, TrajectoryTool, message_groups
from corecoder_mini.delivery import digest, export_artifact, read_json
from corecoder_mini.engine import SUBMIT
from corecoder_mini.events import Events
from corecoder_mini.state_tools import WorkingState
from corecoder_mini.supervisor import verify_artifact
from corecoder_mini.task_state import atomic_json


def large_diagnosis():
    messages = history(12)
    for i in range(12):
        message = messages[3 + i * 2]
        message['content'] = f'PUBLIC_OBSERVATION_{i}\n' + '\n'.join(f'public_source_{i}_{n} = {n}' for n in range(150))
        message['extra'].update(returncode=0, action_id=i + 1, observation_kind='execution' if i in {0, 1, 6, 7} else 'inspection')
    return messages


@pytest.mark.parametrize('recover', [False, True])
def test_preferred_diagnosis_and_recent_windows_archive_under_pressure_without_losing_pairs(tmp_path, recover):
    messages = large_diagnosis()
    original = copy.deepcopy(messages)
    view = ContextView(tmp_path, WorkingState(tmp_path), Events(tmp_path), target_tokens=2000,
        keep_recent=4, history_output_chars=1000, recovery=(lambda: {'reanchor_id': 1}) if recover else None)
    projected, receipt = view.project(messages, 5500)
    assert receipt['capacity_archived_turns'] and receipt['compacted']
    assert estimate_tokens(projected) <= 5500
    assert {11, 12} <= set(receipt['mandatory_turns']) <= set(receipt['retained_turns'])
    _, raw_groups = message_groups(messages)
    _, kept_groups = message_groups(projected)
    for turn, group in zip(receipt['retained_turns'], kept_groups):
        assert group[0]['reasoning_content'] == raw_groups[turn - 1][0]['reasoning_content']
    assert messages == original == read_json(tmp_path / 'canonical-messages.json')
    archived = receipt['capacity_archived_turns'][0]
    assert archived not in receipt['retained_turns']
    recovered = json.loads(TrajectoryTool(tmp_path).execute(turn=archived, offset=0, limit=4000))
    assert recovered[0]['observations'][0]['content'] == raw_groups[archived - 1][1]['content'][:4000]
    assert 'trajectory_read' in json.dumps(projected) and 'Archived turns' in json.dumps(projected)
    if recover:
        assert read_json(tmp_path / 'private/action-recovery/0001.json')['history_pruned']
    later, later_receipt = view.project(messages, 5500)
    assert estimate_tokens(later) <= 5500 and later_receipt['capacity_archived_turns']
    message_groups(later)


@pytest.mark.parametrize('recover', [False, True])
def test_actual_latest_edit_check_and_failure_remain_exact_during_capacity_fallback(tmp_path, recover):
    messages = large_diagnosis()
    messages[9]['extra']['workspace_changed'] = True  # Turn 4 is the last mutation.
    messages[13]['extra'].update(returncode=1, verification_candidate=True)  # Turn 6 failed.
    messages[15]['extra']['verification_candidate'] = True  # A later different check is not a waiver.
    view = ContextView(tmp_path, WorkingState(tmp_path), Events(tmp_path), target_tokens=2000,
        history_output_chars=1000, recovery=(lambda: {'reanchor_id': 1}) if recover else None)
    projected, receipt = view.project(messages, 7800)
    assert receipt['capacity_archived_turns'] and {4, 6, 7, 11, 12} <= set(receipt['mandatory_turns'])
    _, groups = message_groups(projected)
    retained = dict(zip(receipt['retained_turns'], groups))
    for turn in receipt['mandatory_turns']:
        original_group = message_groups(messages)[1][turn - 1]
        assert retained[turn][0]['tool_calls'] == original_group[0]['tool_calls']
        assert retained[turn][1]['content'] == original_group[1]['content']
    assert estimate_tokens(projected) <= 7800


def test_unfittable_mandatory_failure_stops_before_transport_with_durable_numeric_diagnosis(tmp_path):
    messages = large_diagnosis()
    messages[-1]['extra'].update(returncode=1, verification_candidate=True)
    messages[-1]['content'] = 'ACTUAL_FAILURE\n' + 'unresolved assertion detail\n' * 1200
    original = copy.deepcopy(messages)
    view = ContextView(tmp_path, WorkingState(tmp_path), Events(tmp_path), target_tokens=2000, history_output_chars=1000)
    with pytest.raises(RuntimeError, match='context_capacity'):
        view.project(messages, 5000)
    diagnosis = read_json(tmp_path / 'context-capacity.json')
    assert diagnosis['status'] == 'blocked' and diagnosis['required_tokens'] > diagnosis['capacity']
    assert not diagnosis['transport_started'] and 12 in diagnosis['mandatory_turns']
    assert messages == original == read_json(tmp_path / 'canonical-messages.json')
    assert 'ACTUAL_FAILURE' not in json.dumps(diagnosis)


@pytest.mark.docker
def test_long_native_investigation_can_project_then_edit_verify_submit_and_accept_patch(tmp_path, monkeypatch):
    create_case(tmp_path)
    for n in range(6):
        for folder in ['baseline', 'workspace']:
            text = '\n'.join(f'public_context_{n}_{i:04d} = {i:04d}' for i in range(500))
            (tmp_path / folder / f'context{n}.txt').write_text(text)
    from scripts.swebench_io import tree_hashes
    frozen = read_json(tmp_path / 'frozen.json')
    frozen['baseline_sha256'] = tree_hashes(tmp_path / 'baseline')
    atomic_json(tmp_path / 'frozen.json', frozen)
    check = "python -c 'from calc import add; assert add(2,3)==5; assert add(-1,1)==0'"
    steps = [('bash', {'command': f'python -c "from pathlib import Path; print(Path(\'context{n}.txt\').read_text())"'}) for n in range(6)]
    steps += [('edit_file', {'file_path': 'calc.py', 'old_string': 'a - b', 'new_string': 'a + b'}),
              ('bash', {'command': check}), ('bash', {'command': SUBMIT})]
    with local_provider(lambda data, n: (200, response([steps[n - 1]]))) as (url, requests):
        agent = make_agent(tmp_path, monkeypatch, url, extensions=['trajectory_read'],
            budget=Budget(.1, 300000, 14), context_tokens=14000, history_target_tokens=4000)
        assert agent.run('Fix add, check both signs and submit.')['exit_status'] == 'Submitted'
        assert len(requests) == 9
        receipts = [read_json(f) for f in (tmp_path / 'agent/requests').glob('*.json')]
        assert any(r['projection'].get('capacity_archived_turns') for r in receipts)
        for r in receipts:
            assert r['status'] == 'returned'
    exported = export_artifact(tmp_path, {'reason': 'public scripted capacity development fixture'})
    assert exported['model_submitted'] and exported['patch_bytes'] > 0
    accepted = verify_artifact(tmp_path)
    assert accepted['autonomous_complete'] and accepted['official_correct'] is None
    assert digest(tmp_path / 'model.patch') == accepted['patch_sha256']
