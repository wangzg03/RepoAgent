"""Wire-level compatibility checks against localhost, without paid inference."""

import pytest
from test_native_upstream import local_provider, make_agent, response

from corecoder_mini.budget import Budget
from corecoder_mini.provider_parameters import completion_parameters
from corecoder_mini.supervisor import validate_extensions


@pytest.mark.parametrize('extensions', [[], ['scratchpad', 'trajectory_read']])
def test_qwen37_parameters_reach_actual_native_http_and_total_output_reservation(tmp_path, monkeypatch, extensions):
    def reply(data, number):
        assert data['temperature'] == .6
        assert data['max_completion_tokens'] == 8192 and 'max_tokens' not in data
        assert data['enable_thinking'] is True and data['preserve_thinking'] is True
        assert data['thinking_budget'] == 2048 and 'reasoning_effort' not in data
        return 200, response([('bash', {'command': 'echo public fixture'})])
    with local_provider(reply) as (url, requests):
        agent = make_agent(tmp_path, monkeypatch, url, extensions=extensions,
                           max_output_tokens=8192, context_tokens=32000, output_limit_parameter='max_completion_tokens',
                           temperature=.6, enable_thinking=True, preserve_thinking=True, thinking_budget=2048)
        agent.model.query([{'role': 'user', 'content': 'Public development task'}])
        assert len(requests) == agent.model.provider.budget.requests == 1
    params = {'messages': [], 'max_completion_tokens': 8192}
    assert Budget.quote(params, (0, 1)).tokens >= 8192


@pytest.mark.parametrize('settings', [
    {'temperature': True}, {'temperature': float('nan')}, {'temperature': 3},
    {'output_limit_parameter': 'made_up_cap'}, {'thinking_budget': 0}, {'thinking_budget': True},
    {'enable_thinking': 'true'}, {'preserve_thinking': 1},
    {'thinking_budget': 100, 'reasoning_effort': 'low'},
    {'thinking_budget': 100, 'enable_thinking': False},
    {'thinking_budget': 1024, 'max_output_tokens': 1024, 'output_limit_parameter': 'max_completion_tokens'},
])
def test_invalid_provider_parameters_stop_before_freeze_or_http(settings):
    with pytest.raises(ValueError):
        validate_extensions(settings)


def test_old_config_keeps_its_original_sampling_and_output_parameter():
    assert completion_parameters({'max_output_tokens': 1024}) == {'max_tokens': 1024, 'temperature': 0}
