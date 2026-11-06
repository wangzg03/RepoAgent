"""Real localhost SSE plus Docker; no external model or benchmark answers."""

import json
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from test_loop import create_case
from test_native_upstream import local_provider, make_agent, response

from corecoder_mini.budget import Budget
from corecoder_mini.delivery import export_artifact, read_json
from corecoder_mini.engine import SUBMIT
from corecoder_mini.native_model import ProviderFailure
from corecoder_mini.runtime import Cancelled, interruptible_call, local
from corecoder_mini.supervisor import validate_extensions, verify_artifact
from corecoder_mini.task_state import atomic_json
from corecoder_mini.upstream import FormatError


@pytest.fixture(autouse=True)
def clear_runtime():
    vars(local).clear()
    yield
    vars(local).clear()


def chunk(delta=None, finish=None, usage=None):
    item = {'id': 'sse-public-fixture', 'object': 'chat.completion.chunk', 'created': 1, 'model': 'fixture',
            'choices': [{'index': 0, 'delta': delta or {}, 'finish_reason': finish}]}
    if usage is not None:
        item.update(choices=[], usage=usage)
    return item


USAGE = {'prompt_tokens': 50, 'completion_tokens': 20, 'total_tokens': 70,
         'prompt_tokens_details': {'cached_tokens': 30}}


def fragments(calls, *, finish='tool_calls', usage=True, done=True):
    events = [chunk({'role': 'assistant', 'reasoning_content': 'opaque-'}),
              chunk({'reasoning_content': 'native-state', 'content': 'Fixture'})]
    for index, (name, args) in enumerate(calls):
        arguments = json.dumps(args, ensure_ascii=False)
        split = len(arguments) // 2
        events.append(chunk({'tool_calls': [{'index': index, 'id': f'sse-call-{index}', 'type': 'function',
            'function': {'name': name, 'arguments': arguments[:split]}}]}))
        events.append(chunk({'tool_calls': [{'index': index, 'function': {'arguments': arguments[split:]}}]}))
    if finish is not None:
        events.append(chunk(finish=finish))
    if usage:
        events.append(chunk(usage=USAGE))
    if done:
        events.append('DONE')
    return events


@contextmanager
def sse_provider(callback):
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append(data)
            status, events = callback(data, len(requests))
            self.send_response(status)
            if status != 200:
                body = json.dumps(events).encode()
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            try:
                for event in events:
                    if isinstance(event, (int, float)):
                        time.sleep(event)
                        continue
                    body = '[DONE]' if event == 'DONE' else json.dumps(event, ensure_ascii=False)
                    self.wfile.write(('data: ' + body + '\n\n').encode())
                    self.wfile.flush()
            except (OSError, BrokenPipeError):
                pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/v1', requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


@pytest.mark.parametrize('extensions', [[], ['trajectory_read']])
def test_sse_fragmented_parallel_calls_reasoning_cache_and_next_request(tmp_path, monkeypatch, extensions):
    calls = [('bash', {'command': 'echo first'}), ('bash', {'command': 'echo "second\\nline"'})]
    def answer(data, number):
        assert data['stream'] is True and data['stream_options'] == {'include_usage': True}
        assert data['reasoning_effort'] == 'medium' and data['temperature'] == .6
        assert data['max_completion_tokens'] == 8192 and 'thinking_budget' not in data
        assert data['preserve_thinking'] is True
        if number == 2:
            previous = next(m for m in data['messages'] if m['role'] == 'assistant')
            assert previous['reasoning_content'] == 'opaque-native-state'
            assert [m['tool_call_id'] for m in data['messages'] if m['role'] == 'tool'] == ['sse-call-0', 'sse-call-1']
        return 200, fragments(calls)
    with sse_provider(answer) as (url, requests):
        agent = make_agent(tmp_path, monkeypatch, url, extensions=extensions, provider_attempts=1,
            output_limit_parameter='max_completion_tokens', max_output_tokens=8192, context_tokens=32000,
            reasoning_effort='medium', temperature=.6, enable_thinking=True, preserve_thinking=True)
        agent.model.provider.pricing = (1, 2, .1)
        messages = [{'role': 'user', 'content': 'Public fixture'}]
        reply = agent.model.query(messages)
        assert [a.get('arguments', {'command': a.get('command')}) for a in reply['extra']['actions']] == [c[1] for c in calls]
        observed = agent.model.format_observation_messages(reply, [{'returncode': 0, 'output': 'fixture', 'exception_info': ''}] * 2)
        agent.model.query(messages + [reply] + observed)
        assert len(requests) == agent.model.provider.budget.requests == 2
    receipt = read_json(tmp_path / 'agent/requests/0001.json')
    assert receipt['stream']['completed'] and receipt['stream']['chunks'] == 8
    assert receipt['stream']['first_chunk_seconds'] >= 0
    assert receipt['prompt_tokens'] == 50 and receipt['cached_prompt_tokens'] == 30
    assert receipt['cost'] == pytest.approx((20 + 30 * .1 + 20 * 2) / 1e6)


@pytest.mark.parametrize('finish,done,expected', [(None, False, ProviderFailure), (None, True, ProviderFailure),
                                              ('length', True, FormatError), ('content_filter', True, FormatError)])
def test_incomplete_or_length_response_never_executes_even_valid_tool_json(tmp_path, monkeypatch, finish, done, expected):
    events = fragments([('bash', {'command': 'touch must-not-exist'})], finish=finish, usage=False, done=done)
    with sse_provider(lambda *args: (200, events)) as (url, requests):
        agent = make_agent(tmp_path, monkeypatch, url, extensions=['trajectory_read'], provider_attempts=1)
        agent.add_messages({'role': 'user', 'content': 'Public fixture'})
        with pytest.raises(expected):
            agent.step()
        assert len(requests) == 1 and not (tmp_path / 'agent/actions').exists()
        assert not (tmp_path / 'workspace/must-not-exist').exists()
        assert agent.env.registry.tools['bash'].sandbox.environment is None


def test_missing_provider_usage_keeps_reservation_despite_litellm_synthetic_usage(tmp_path, monkeypatch):
    with sse_provider(lambda *args: (200, fragments([('bash', {'command': 'echo fixture'})], usage=False))) as (url, _):
        agent = make_agent(tmp_path, monkeypatch, url, provider_attempts=1)
        agent.model.query([{'role': 'user', 'content': 'Public fixture'}])
    receipt = read_json(tmp_path / 'agent/requests/0001.json')
    assert receipt['status'] == 'returned' and receipt['stream']['completed']
    assert not receipt['usage_complete'] and not receipt['stream']['usage_received']
    assert receipt['cost_basis'] == 'unknown_usage_reservation' and receipt['prompt_tokens'] == 0
    assert agent.model.provider.budget.unknown_usage_requests == 1


def test_usage_is_settled_even_when_the_response_has_no_terminal_event(tmp_path, monkeypatch):
    events = fragments([('bash', {'command': 'echo fixture'})], finish=None)
    with sse_provider(lambda *args: (200, events)) as (url, _):
        agent = make_agent(tmp_path, monkeypatch, url, provider_attempts=1)
        with pytest.raises(ProviderFailure):
            agent.model.query([{'role': 'user', 'content': 'Public fixture'}])
    receipt = read_json(tmp_path / 'agent/requests/0001.json')
    assert receipt['usage_complete'] and receipt['prompt_tokens'] == 50
    assert receipt['stream']['usage_received'] and not receipt['stream']['completed']
    assert agent.model.provider.budget.unknown_usage_requests == 0


def test_oversized_wire_stream_stops_without_executing_or_retrying(tmp_path, monkeypatch):
    with sse_provider(lambda *args: (200, [chunk({'content': 'x' * (1024 * 1024 + 1)})])) as (url, requests):
        agent = make_agent(tmp_path, monkeypatch, url, provider_attempts=2)
        with pytest.raises(ProviderFailure):
            agent.model.query([{'role': 'user', 'content': 'Public fixture'}])
        assert len(requests) == 1 and not (tmp_path / 'agent/actions').exists()
    receipt = read_json(tmp_path / 'agent/requests/0001.json')
    assert receipt['diagnostic']['category'] == 'response_bound'
    assert receipt['stream']['response_bytes'] > 1024 * 1024


def test_many_small_valid_sdk_events_do_not_consume_the_model_content_limit_with_metadata(tmp_path, monkeypatch):
    events = [chunk({'reasoning_content': 'x'}) for _ in range(2600)]
    usage = {**USAGE, 'completion_tokens': 2610, 'total_tokens': 2660}
    events += [chunk({'tool_calls': [{'index': 0, 'id': 'one', 'type': 'function',
        'function': {'name': 'bash', 'arguments': '{"command":"echo fixture"}'}}]}),
        chunk(finish='tool_calls'), chunk(usage=usage), 'DONE']
    with sse_provider(lambda *args: (200, events)) as (url, requests):
        agent = make_agent(tmp_path, monkeypatch, url, provider_attempts=1, max_output_tokens=4096, context_tokens=32000)
        reply = agent.model.query([{'role': 'user', 'content': 'Public fixture'}])
        assert len(requests) == 1 and reply['reasoning_content'] == 'x' * 2600
        assert reply['extra']['actions'][0]['command'] == 'echo fixture'
    receipt = read_json(tmp_path / 'agent/requests/0001.json')
    assert receipt['stream']['response_bytes'] > 1024 * 1024
    assert receipt['stream']['payload_bytes'] < 1024 * 1024 and receipt['stream']['completed']
    assert receipt['usage_complete'] and receipt['completion_tokens'] == 2610


def test_late_opened_transport_is_closed_after_request_timeout():
    released, closed = threading.Event(), threading.Event()
    class Resource:
        def close(self):
            closed.set()
    def late_open():
        released.wait(1)
        return Resource()
    try:
        with pytest.raises(TimeoutError):
            interruptible_call(late_open, timeout=.05)
    finally:
        released.set()
    assert closed.wait(1)


@pytest.mark.parametrize('where', ['headers', 'trickle'])
def test_whole_request_timeout_covers_headers_and_trickling_stream_without_poisoning_task(tmp_path, monkeypatch, where):
    def answer(*args):
        if where == 'headers':
            time.sleep(.6)
        events = [chunk({'content': 'x'}), .08] * 12
        return 200, events
    with sse_provider(answer) as (url, requests):
        agent = make_agent(tmp_path, monkeypatch, url, request_seconds=.25, provider_attempts=1)
        local.cancel = threading.Event()
        started = time.monotonic()
        with pytest.raises(ProviderFailure):
            agent.model.query([{'role': 'user', 'content': 'Public fixture'}])
        assert time.monotonic() - started < .9 and not local.cancel.is_set()
        assert len(requests) <= 1 and not (tmp_path / 'agent/actions').exists()
    receipt = read_json(tmp_path / 'agent/requests/0001.json')
    assert receipt['diagnostic']['category'] == 'timeout' and not receipt['usage_complete']


def test_cancel_after_valid_argument_prefix_discards_late_response(tmp_path, monkeypatch):
    events = fragments([('bash', {'command': 'touch must-not-exist'})])
    events.insert(4, 1)
    with sse_provider(lambda *args: (200, events)) as (url, _):
        agent = make_agent(tmp_path, monkeypatch, url)
        local.cancel = threading.Event()
        timer = threading.Timer(.2, local.cancel.set)
        timer.start()
        agent.add_messages({'role': 'user', 'content': 'Public fixture'})
        started = time.monotonic()
        with pytest.raises(Cancelled):
            agent.step()
        timer.join(timeout=1)
        assert time.monotonic() - started < .8
        assert not (tmp_path / 'agent/actions').exists()
    assert not (tmp_path / 'workspace/must-not-exist').exists()


def test_malformed_complete_tool_arguments_are_not_salvaged(tmp_path, monkeypatch):
    events = [chunk({'tool_calls': [{'index': 0, 'id': 'bad', 'type': 'function',
        'function': {'name': 'bash', 'arguments': '{"command":"touch must-not-exist"'}}]}),
        chunk(finish='tool_calls'), chunk(usage=USAGE), 'DONE']
    with sse_provider(lambda *args: (200, events)) as (url, _):
        agent = make_agent(tmp_path, monkeypatch, url, provider_attempts=1)
        agent.add_messages({'role': 'user', 'content': 'Public fixture'})
        with pytest.raises(FormatError):
            agent.step()
        assert not (tmp_path / 'agent/actions').exists()


def test_nonstream_control_keeps_original_protocol_but_rejects_truncated_tools(tmp_path, monkeypatch):
    def answer(data, number):
        assert data.get('stream', False) is False and 'stream_options' not in data
        return 200, response([('bash', {'command': 'touch must-not-exist'})], finish_reason='length')
    with local_provider(answer) as (url, _):
        agent = make_agent(tmp_path, monkeypatch, url, provider_stream=False, provider_attempts=1)
        agent.add_messages({'role': 'user', 'content': 'Public fixture'})
        with pytest.raises(FormatError):
            agent.step()
        assert not (tmp_path / 'agent/actions').exists()


@pytest.mark.parametrize('value', ['true', 1, None])
def test_stream_setting_is_explicit_boolean(value):
    with pytest.raises(ValueError, match='provider_stream'):
        validate_extensions({'provider_stream': value})


@pytest.mark.docker
@pytest.mark.parametrize('failure_kind', ['524', 'eof'])
def test_sse_recovery_edit_check_submit_and_independent_patch_acceptance(tmp_path, monkeypatch, failure_kind):
    create_case(tmp_path)
    frozen = read_json(tmp_path / 'frozen.json')
    frozen['model']['pricing_usd_per_million'] = [1, 2]
    atomic_json(tmp_path / 'frozen.json', frozen)
    check = "python -c 'from calc import add; assert add(2,3)==5; assert add(-1,1)==0'"
    steps = [('read_file', {'file_path': 'calc.py'}),
             ('edit_file', {'file_path': 'calc.py', 'old_string': 'a - b', 'new_string': 'a + b'}),
             ('bash', {'command': check}), ('bash', {'command': SUBMIT})]
    def answer(data, number):
        assert data['stream'] is True
        if number == 1:
            if failure_kind == 'eof':
                return 200, fragments([('write_file', {'file_path': 'must-not-exist.py', 'content': 'partial = True'})],
                                      finish=None, usage=False, done=False)
            return 524, {'error': {'message': 'public local gateway fixture', 'type': 'server_error'}}
        return 200, fragments([steps[number - 2]])
    with sse_provider(answer) as (url, requests):
        agent = make_agent(tmp_path, monkeypatch, url, extensions=['trajectory_read'],
            budget=Budget(.1, 300000, 12), provider_attempts=2)
        assert agent.run('Fix arithmetic, check both signs and submit.')['exit_status'] == 'Submitted'
        assert len(requests) == agent.model.provider.budget.requests == 5
        assert agent.model.provider.budget.retry_attempts == 1
        assert not (tmp_path / 'workspace/must-not-exist.py').exists()
    result = export_artifact(tmp_path, {'reason': 'scripted localhost SSE fixture'})
    assert result['model_submitted'] and result['provider_recovery']['recovered_provider_calls'] == 1
    assert result['cost_accounting']['unknown_usage_requests'] == 1
    accepted = verify_artifact(tmp_path)
    assert accepted['autonomous_complete'] and accepted['official_correct'] is None
    assert accepted['patch_sha256'] == result['patch_sha256']
    assert read_json(tmp_path / 'agent/actions/0002.json')['workspace_changed']
    assert read_json(tmp_path / 'agent/actions/0003.json')['verification_candidate']
