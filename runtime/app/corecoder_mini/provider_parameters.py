"""Explicit, bounded OpenAI-compatible completion parameters; no model guessing."""

import math


def completion_parameters(settings):
    limit_name = settings.get('output_limit_parameter', 'max_tokens')
    if limit_name not in {'max_tokens', 'max_completion_tokens'}:
        raise ValueError('invalid output_limit_parameter')
    output = settings.get('max_output_tokens', 1024)
    if type(output) is not int or output < 1:
        raise ValueError('max_output_tokens must be a positive integer')
    temperature = settings.get('temperature', 0)
    if type(temperature) not in {int, float} or not math.isfinite(temperature) or not 0 <= temperature <= 2:
        raise ValueError('temperature must be finite and between 0 and 2')
    result = {limit_name: output, 'temperature': temperature}
    body = {}
    for name in ['enable_thinking', 'preserve_thinking']:
        if name in settings:
            if type(settings[name]) is not bool:
                raise ValueError(name + ' must be boolean')
            body[name] = settings[name]
    if 'thinking_budget' in settings:
        thinking = settings['thinking_budget']
        if type(thinking) is not int or not 1 <= thinking <= 262144:
            raise ValueError('thinking_budget must be a positive integer <= 262144')
        if settings.get('reasoning_effort') or body.get('enable_thinking') is False:
            raise ValueError('thinking_budget conflicts with reasoning_effort or disabled thinking')
        if limit_name == 'max_completion_tokens' and thinking >= output:
            raise ValueError('thinking_budget must leave room for completion inside total output cap')
        body['thinking_budget'] = thinking
    if body:
        result['extra_body'] = body
    allowed = ['max_completion_tokens'] if limit_name == 'max_completion_tokens' else []
    if settings.get('reasoning_effort'):
        result['reasoning_effort'] = settings['reasoning_effort']
        allowed.append('reasoning_effort')
    if allowed:
        result['allowed_openai_params'] = allowed
    return result
