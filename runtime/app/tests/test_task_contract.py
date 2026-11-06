"""Declared task goals drive guidance, without model guesses or new tool gates."""

import json
from types import SimpleNamespace

import pytest
from test_harness import ScriptedProvider
from test_loop import IMAGE, PREFIX, create_case

from corecoder_mini import supervisor
from corecoder_mini.continuation import ActionContinuation
from corecoder_mini.delivery import export_artifact
from corecoder_mini.engine import SUBMIT, build_agent
from corecoder_mini.progress import WorkspaceProgress
from corecoder_mini.state_tools import TodoTool, WorkingState


def test_declared_contract_survives_restart_and_model_summary_cannot_change_it(tmp_path):
    state = WorkingState(tmp_path)
    state.bind_goal('Fix the public task')
    state.bind_task_contract('fix', 'swebench_task_definition')
    revision = state.data['revision']
    state.bind_task_contract('fix', 'swebench_task_definition')
    assert state.data['revision'] == revision
    state.update(lambda d: d.update(intent={'kind': 'review', 'requirements': []}))
    restored = WorkingState(tmp_path)
    assert restored.requested_kind() == 'fix'
    assert restored.view()['task_contract'] == {'kind': 'fix', 'source': 'swebench_task_definition'}
    with pytest.raises(ValueError, match='changed_on_restart'):
        restored.bind_task_contract('review')
    assert restored.requested_kind() == 'fix'


def test_unknown_user_intent_is_not_guessed_or_automatically_marked_complete(tmp_path):
    state = WorkingState(tmp_path)
    state.bind_goal('Discuss this request')
    assert state.requested_kind() is None and 'task_contract' not in state.view()
    assert state.data['notes'] == {} and state.data['todos'] == []
    assert TodoTool(state).check()['acceptance'] is False


@pytest.mark.parametrize('kind', ['fix', 'implement', 'refactor', 'test', 'explain', 'locate', 'review'])
def test_runtime_renders_the_declared_work_type_without_extra_planning_calls(tmp_path, kind):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    provider = ScriptedProvider([])
    agent = build_agent(provider, workspace, tmp_path / 'agent', IMAGE,
                        {**supervisor.DEFAULTS, 'harness_tools': ['trajectory_read'], 'task_intent': kind})
    agent.extra_template_vars['task'] = 'Original public requirement'
    text = agent._render_template(agent.config.instance_template)
    assert 'Original public requirement' in text and 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT' in text
    assert not provider.requests
    if kind in {'explain', 'locate', 'review'}:
        assert 'Do not modify source' in text and 'make a scoped change' not in text
    else:
        assert 'make a scoped change' in text and 'repair failures and rerun' in text
        assert 'For read-only work, deliver findings' not in text
    assert {'bash', 'read_file', 'edit_file', 'write_file'} <= agent.env.registry.tools.keys()


def test_swebench_freezes_fix_contract_before_inference_without_reading_judge(tmp_path, monkeypatch):
    workspace = tmp_path / 'source'
    workspace.mkdir()
    (workspace / 'calc.py').write_text('value = 1\n')
    calls = []
    monkeypatch.setattr(supervisor.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout=IMAGE))
    monkeypatch.setattr('corecoder_mini.swebench.prepare', lambda directory, config, task:
                        calls.append(task) or {'kind': 'swebench_official'})
    config = {'workspace': str(workspace), 'task_id': 'public-fixture', 'task': 'Original issue text',
              'provider': 'openai', 'model': 'fixture', 'base_url': 'http://127.0.0.1:12345',
              'pricing_usd_per_million': [1, 2], 'image': IMAGE, 'verification': {'kind': 'swebench_official'}}
    directory = supervisor.freeze(config, tmp_path / 'run')
    frozen = json.loads((directory / 'frozen.json').read_text())
    assert frozen['settings']['task_intent'] == 'fix'
    assert frozen['settings']['task_intent_source'] == 'swebench_task_definition'
    assert frozen['settings']['task_kind'] == 'benchmark' and not frozen['learning_enabled']
    assert frozen['public_task'] == 'Original issue text' and calls == ['public-fixture']


def test_official_fix_contract_does_not_silently_override_declared_read_only_work(tmp_path):
    with pytest.raises(ValueError, match='conflicts_with_declared'):
        supervisor.freeze({'settings': {'task_intent': 'review'}, 'verification': {'kind': 'swebench_official'}}, tmp_path / 'run')
    assert not (tmp_path / 'run').exists()


def test_guidance_uses_trusted_fix_contract_even_with_empty_todo(tmp_path):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    (workspace / 'calc.py').write_text('value = 1\n')
    state = WorkingState(tmp_path / 'agent')
    state.bind_task_contract('fix', 'swebench_task_definition')
    state.update(lambda d: d.update(intent={'kind': 'review'}))
    progress = WorkspaceProgress(workspace, tmp_path / 'agent', {'stall_request_limit': 1}, state)
    progress.observe(1, 'bash', {'command': 'cat calc.py'}, {'returncode': 0, 'output': 'value = 1'})
    packet = ActionContinuation(progress, SimpleNamespace(tools={'bash': None, 'edit_file': None}))()
    assert packet['facts']['intent'] == 'fix' and packet['facts']['intent_source'] == 'swebench_task_definition'
    assert 'For read-only work' not in packet['instruction']
    assert not progress.request_feedback()['read_only_intent']
    assert packet['acceptance'] is False and state.data['todos'] == []


@pytest.mark.parametrize('value', ['unknown', [], 1, True])
def test_invalid_operator_intent_is_rejected_before_transport(value):
    with pytest.raises(ValueError, match='task_intent'):
        supervisor.validate_extensions({'task_intent': value})


@pytest.mark.docker
def test_declared_fix_with_no_todo_really_edits_checks_submits_and_accepts_same_patch(tmp_path):
    create_case(tmp_path)
    check = "python -c 'from calc import add; assert add(2,3)==5; assert add(-1,1)==0'"
    provider = ScriptedProvider([('read_file', {'file_path': 'calc.py'}),
        ('edit_file', {'file_path': 'calc.py', 'old_string': 'return a - b', 'new_string': 'return a + b'}),
        ('bash', {'command': check}), ('bash', {'command': SUBMIT})])
    agent = build_agent(provider, tmp_path / 'workspace', tmp_path / 'agent', IMAGE,
                        {**supervisor.DEFAULTS, 'shell_prefix': PREFIX, 'task_intent': 'fix',
                         'task_intent_source': 'swebench_task_definition', 'harness_tools': ['trajectory_read']})
    assert agent.run('Original public issue: addition is incorrect.')['exit_status'] == 'Submitted'
    assert len(provider.requests) == 4
    assert agent.working_state.requested_kind() == 'fix' and not agent.working_state.data['todos']
    export_artifact(tmp_path, {'reason': 'public scripted development fixture'})
    assert supervisor.verify_artifact(tmp_path)['autonomous_complete']
