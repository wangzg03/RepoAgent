"""Structured editing integration: real sandbox effects, no paid model requests."""

import hashlib
import json
import shlex
import threading
from types import SimpleNamespace

import pytest
from test_harness import ScriptedProvider
from test_loop import IMAGE, PREFIX, create_case
from test_native_upstream import local_provider, make_agent, response

from corecoder_mini.budget import Budget
from corecoder_mini.delivery import digest, export_artifact, read_json
from corecoder_mini.engine import SUBMIT, build_agent
from corecoder_mini.runtime import Cancelled, local
from corecoder_mini.supervisor import DEFAULTS, validate_extensions, verify_artifact
from corecoder_mini.task_state import atomic_json
from corecoder_mini.tools.registry import ToolRegistry
from corecoder_mini.tools.workspace import EditFileTool, WriteFileTool
from scripts.swebench_io import tree_hashes


@pytest.fixture(autouse=True)
def clear_runtime():
    vars(local).clear()
    yield
    vars(local).clear()


@pytest.fixture
def file_agent(tmp_path):
    create_case(tmp_path)
    agent = build_agent(ScriptedProvider([]), tmp_path / 'workspace', tmp_path / 'agent', IMAGE,
                        {**DEFAULTS, 'shell_prefix': PREFIX, 'harness_tools': ['trajectory_read']})
    yield agent
    agent.env.cleanup()


def invoke(agent, name, **arguments):
    result = agent.env.execute({'tool': name, 'arguments': arguments, 'tool_call_id': 'file-tool-fixture'})
    return result, json.loads(result['output'])


@pytest.mark.parametrize('path', ['../outside.py', '/etc/passwd', 'src/../../outside.py',
    '.git/config', '/workspace/.git/config', 'src/.git./config', 'src/.. /outside.py',
    'C:/outside.py', 'src\\file.py', '.env', '.env.production', 'key.txt', '.pytest_cache/private', ''])
def test_invalid_paths_rejected_before_transport(path):
    class Forbidden:
        def run(self, *args, **kwargs):
            pytest.fail('Invalid input must not reach execution')
    registry = ToolRegistry([EditFileTool(Forbidden(), 10)])
    with pytest.raises(ValueError):
        registry.validate('edit_file', {'file_path': path, 'old_string': 'old', 'new_string': 'new'})


@pytest.mark.parametrize('arguments', [
    {'old_string': '', 'new_string': 'new'}, {'old_string': 'old', 'new_string': 'old'},
    {'old_string': 'old', 'new_string': 'new', 'expected_sha256': 'x' * 64},
    {'old_string': 'old', 'new_string': 'new', 'replace_all': True},
    {'old_string': 'old\0', 'new_string': 'new'}, {'old_string': 'old', 'new_string': 'new\0'},
])
def test_invalid_edit_contract_rejected_before_transport(arguments):
    registry = ToolRegistry([EditFileTool(SimpleNamespace(), 10)])
    with pytest.raises((ValueError, TypeError)):
        registry.validate('edit_file', {'file_path': 'calc.py', **arguments})


def test_binary_new_content_rejected_before_transport():
    registry = ToolRegistry([WriteFileTool(SimpleNamespace(), 10)])
    with pytest.raises(ValueError):
        registry.validate('write_file', {'file_path': 'code.py', 'content': 'binary\0payload'})


def test_standard_and_child_actual_tool_catalogs_and_bash_only_control(tmp_path):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    for depth, expected in [(0, {'read_file', 'edit_file', 'write_file'}), (1, {'read_file'})]:
        agent = build_agent(ScriptedProvider([]), workspace, tmp_path / str(depth), IMAGE,
                            {**DEFAULTS, 'harness_tools': ['trajectory_read']}, depth=depth)
        names = set(agent.env.registry.tools)
        assert names & {'read_file', 'edit_file', 'write_file'} == expected
        assert all(name in agent.config.instance_template for name in expected)
    control = build_agent(ScriptedProvider([]), workspace, tmp_path / 'control', IMAGE, {**DEFAULTS, 'harness_tools': []})
    assert list(control.env.registry.tools) == ['bash']
    settings = {'harness_tools': ['edit_file']}
    validate_extensions(settings)
    explicit = build_agent(ScriptedProvider([]), workspace, tmp_path / 'explicit', IMAGE, {**DEFAULTS, **settings})
    assert 'edit_file' in explicit.env.registry.tools


def test_file_cancellation_persists_ambiguous_intent_and_never_falls_back_to_host(file_agent, tmp_path):
    tool = file_agent.env.registry.tools['edit_file']
    calls = []
    def interrupted(*args, **kwargs):
        calls.append(1)
        assert read_json(tmp_path / 'agent/actions/0001.json')['status'] == 'intent'
        raise Cancelled('Public fixture interruption')
    tool.sandbox = SimpleNamespace(run=interrupted, root=tmp_path / 'workspace', read_only=False)
    before = (tmp_path / 'workspace/calc.py').read_bytes()
    with pytest.raises(Cancelled):
        file_agent.env.execute({'tool': 'edit_file', 'arguments': {'file_path': 'calc.py', 'old_string': 'a - b', 'new_string': 'a + b'}})
    assert (tmp_path / 'workspace/calc.py').read_bytes() == before and calls == [1]
    assert read_json(tmp_path / 'agent/actions/0001.json')['status'] == 'interrupted_or_failed'
    # Constructing a new task does not replay an ambiguous file operation.
    build_agent(ScriptedProvider([]), tmp_path / 'workspace', tmp_path / 'new-agent', IMAGE, DEFAULTS)
    assert calls == [1]


def test_cancelled_file_edit_stops_before_transport(file_agent, monkeypatch):
    local.cancel = threading.Event()
    local.cancel.set()
    monkeypatch.setattr(file_agent.env.registry.tools['edit_file'].sandbox, 'run',
                        lambda *a, **k: pytest.fail('Cancelled task reached execution'))
    with pytest.raises(Cancelled):
        invoke(file_agent, 'edit_file', file_path='calc.py', old_string='a - b', new_string='a + b')


@pytest.mark.docker
def test_exact_edit_hash_conflict_and_error_repair_receipts(file_agent, tmp_path):
    result, read = invoke(file_agent, 'read_file', file_path='calc.py')
    assert result['returncode'] == 0 and 'return a - b' in read['content']
    assert read['sha256'] == digest(tmp_path / 'workspace/calc.py')
    result, missing = invoke(file_agent, 'edit_file', file_path='calc.py', old_string='not present', new_string='replacement')
    assert result['returncode'] == 2 and missing['error'] == 'anchor_not_found'
    assert digest(tmp_path / 'workspace/calc.py') == read['sha256']
    result, edit = invoke(file_agent, 'edit_file', file_path='calc.py', old_string='a - b', new_string='a + b', expected_sha256=read['sha256'])
    assert result['returncode'] == 0 and edit['changed'] and edit['before_sha256'] == read['sha256']
    assert edit['after_sha256'] == digest(tmp_path / 'workspace/calc.py') and '+    return a + b' in edit['diff']
    assert not file_agent.env.progress.data['unresolved_commands']
    result, stale = invoke(file_agent, 'edit_file', file_path='calc.py', old_string='a + b', new_string='a * b', expected_sha256=read['sha256'])
    assert result['returncode'] == 2 and stale['error'] == 'version_conflict'
    assert digest(tmp_path / 'workspace/calc.py') == edit['after_sha256']
    receipt = read_json(tmp_path / 'agent/actions/0003.json')
    assert receipt['workspace_changed'] and not receipt['verification_candidate'] and receipt['source_targets'] == ['calc.py']


@pytest.mark.docker
def test_read_pagination_exact_bytes_and_bom_crlf_edit(file_agent, tmp_path):
    path = tmp_path / 'workspace/long.py'
    original = b'\xef\xbb\xbf' + ('value = "' + 'x' * 9000 + '"\r\nreturn_value = 1\r\n').encode()
    path.write_bytes(original)
    text, offset, column = '', 1, 1
    for _ in range(8):
        result, page = invoke(file_agent, 'read_file', file_path='long.py', offset=offset, column=column)
        assert result['returncode'] == 0 and len(result['output']) < 6000
        assert page['sha256'] == hashlib.sha256(original).hexdigest()
        text += page['content']
        if page['next_offset'] is None:
            break
        offset, column = page['next_offset'], page['next_column']
    assert text == original.decode('utf-8-sig')
    result, _ = invoke(file_agent, 'edit_file', file_path='long.py', old_string='return_value = 1\n', new_string='return_value = 2\n')
    assert result['returncode'] == 0
    assert path.read_bytes() == original.replace(b'return_value = 1', b'return_value = 2')


@pytest.mark.docker
def test_ambiguous_edit_overwrite_guard_and_text_limits(file_agent, tmp_path):
    path = tmp_path / 'workspace/repeated.py'
    path.write_text('x = 1\nx = 1\n', encoding='utf-8')
    result, failure = invoke(file_agent, 'edit_file', file_path='repeated.py', old_string='x = 1', new_string='x = 2')
    assert result['returncode'] == 2 and failure['error'] == 'ambiguous_anchor'
    overlap = tmp_path / 'workspace/overlap.py'
    overlap.write_text('ababa', encoding='utf-8')
    result, failure = invoke(file_agent, 'edit_file', file_path='overlap.py', old_string='aba', new_string='xyz')
    assert result['returncode'] == 2 and failure['error'] == 'ambiguous_anchor' and overlap.read_text() == 'ababa'
    result, failure = invoke(file_agent, 'write_file', file_path='repeated.py', content='replacement\n')
    assert result['returncode'] == 2 and failure['error'] == 'expected_hash_required'
    assert path.read_text() == 'x = 1\nx = 1\n'
    result, replaced = invoke(file_agent, 'write_file', file_path='repeated.py', content='replacement\n', expected_sha256=digest(path))
    assert result['returncode'] == 0 and replaced['changed']
    (tmp_path / 'workspace/binary').write_bytes(b'\0bad')
    result, failure = invoke(file_agent, 'read_file', file_path='binary')
    assert result['returncode'] == 2 and failure['error'] == 'unsupported_file'
    (tmp_path / 'workspace/oversized').write_bytes(b'a' * (2 * 2**20 + 1))
    result, failure = invoke(file_agent, 'read_file', file_path='oversized')
    assert result['returncode'] == 2 and failure['error'] == 'unsupported_file'
    result, _ = invoke(file_agent, 'write_file', file_path='marker.txt', content=SUBMIT)
    assert result['returncode'] == 0 and not (tmp_path / 'agent/submission.json').exists()


@pytest.mark.docker
def test_killed_precommit_worker_keeps_original_and_orphan_out_of_patch(file_agent, tmp_path):
    tool = file_agent.env.registry.tools['edit_file']
    # Inject a real SIGKILL at the OS rename boundary, after staging bytes. No
    # production test hook or benchmark patch is introduced.
    program = ("import os, signal\nnamespace={'__name__':'public_io_fixture'}\n"
               f"exec({tool.program!r}, namespace)\n"
               "def kill_at_commit(*args, **kwargs): os.kill(os.getpid(), signal.SIGKILL)\n"
               "os.replace=kill_at_commit\nnamespace['main']()\n")
    before = (tmp_path / 'workspace/calc.py').read_bytes()
    result = tool.sandbox.run(PREFIX + 'python -I -c ' + shlex.quote(program), tool.sandbox.root, 20,
                              input_text=json.dumps({'operation': 'edit_file', 'arguments': {
                                  'file_path': 'calc.py', 'old_string': 'a - b', 'new_string': 'a + b'}}),
                              submission_requested=False)
    assert result.returncode != 0 and (tmp_path / 'workspace/calc.py').read_bytes() == before
    assert list((tmp_path / 'workspace/.pytest_cache/corecoder-filetools').glob('*.tmp'))
    exported = export_artifact(tmp_path, {'reason': 'public crash injection; no automatic retry'})
    assert exported['patch_bytes'] == 0 and not exported['model_submitted']
    result, current = invoke(file_agent, 'read_file', file_path='calc.py')
    assert result['returncode'] == 0 and current['sha256'] == hashlib.sha256(before).hexdigest()


@pytest.mark.docker
def test_symlink_and_fifo_access_is_rejected_inside_linux_without_host_fallback(file_agent):
    sandbox = file_agent.env.registry.tools['bash'].sandbox
    setup = sandbox.run(PREFIX + "printf original > /tmp/outside-source; ln -s /tmp/outside-source alias.py; mkfifo pipe-source", sandbox.root, 10)
    assert setup.returncode == 0
    try:
        for name, args in [('read_file', {'file_path': 'alias.py'}),
                           ('edit_file', {'file_path': 'alias.py', 'old_string': 'original', 'new_string': 'changed'}),
                           ('read_file', {'file_path': 'pipe-source'})]:
            # The monitor deliberately rejects symlinked workspace snapshots;
            # exercise the file boundary directly without asking it to hash one.
            result = file_agent.env.registry.tools[name].execute(**args)
            assert result['returncode'] == 2
        assert sandbox.run('cat /tmp/outside-source', sandbox.root, 10).stdout.strip() == 'original'
    finally:
        sandbox.run('rm alias.py pipe-source', sandbox.root, 10)


@pytest.mark.docker
def test_native_edit_failure_check_repair_submit_and_independent_verification(tmp_path, monkeypatch):
    create_case(tmp_path)
    hashes = {}
    check = "python -c 'from calc import add; assert add(2,3)==5; assert add(-1,1)==0'"
    def answer(data, number):
        names = {t['function']['name'] for t in data['tools']}
        assert {'read_file', 'edit_file', 'write_file'} <= names
        if number in {2, 4}:
            tool = next(m for m in reversed(data['messages']) if m['role'] == 'tool')
            observed = json.loads(json.loads(tool['content'])['output'])
            hashes[number] = observed.get('sha256', observed.get('after_sha256'))
        if number == 3:
            assert 'anchor_not_found' in json.dumps(data['messages'])
        if number == 5:
            assert 'AssertionError' in json.dumps(data['messages'])
        steps = {
            1: ('read_file', {'file_path': 'calc.py'}),
            2: ('edit_file', {'file_path': 'calc.py', 'old_string': 'wrong anchor', 'new_string': 'a + b'}),
            3: ('edit_file', {'file_path': 'calc.py', 'old_string': 'a - b', 'new_string': 'a * b', 'expected_sha256': hashes.get(2)}),
            4: ('bash', {'command': check}),
            5: ('edit_file', {'file_path': 'calc.py', 'old_string': 'a * b', 'new_string': 'a + b', 'expected_sha256': hashes.get(4)}),
            6: ('bash', {'command': check}), 7: ('bash', {'command': SUBMIT})}
        return 200, response([steps[number]])
    with local_provider(answer) as (url, requests):
        agent = make_agent(tmp_path, monkeypatch, url, extensions=['trajectory_read'], budget=Budget(.1, 300000, 12))
        assert agent.run('Fix addition and verify positive and negative sums.')['exit_status'] == 'Submitted'
        assert len(requests) == 7
    assert read_json(tmp_path / 'agent/actions/0002.json')['result']['returncode'] == 2
    assert read_json(tmp_path / 'agent/actions/0004.json')['result']['returncode'] == 1
    assert read_json(tmp_path / 'agent/actions/0006.json')['verification_candidate']
    assert not agent.env.progress.data['unresolved_commands']
    export_artifact(tmp_path, {'reason': 'local scripted protocol fixture'})
    result = verify_artifact(tmp_path)
    assert result['autonomous_complete'] and result['official_correct'] is None
    assert result['harness_observation']['actual_tool_counts']['edit_file'] == 3
    assert result['harness_observation']['unresolved_write_intents'] == 0
    index = read_json(tmp_path / 'agent/private/recreate-episode/trajectory_index.json')
    assert any(row['tool'] == 'edit_file' and row['step'] == 2 for row in index['errors'])
    assert len(index['file_tool_actions']) == 4


@pytest.mark.docker
def test_new_package_creation_and_empty_marker_survive_real_patch_application(tmp_path, monkeypatch):
    create_case(tmp_path)
    public = {'kind': 'local_command', 'command': PREFIX + "python -c 'from package.calc import square; assert square(3)==9; assert square(-2)==4'"}
    atomic_json(tmp_path / 'verification-config.json', public)
    frozen = read_json(tmp_path / 'frozen.json')
    for folder in ['baseline', 'workspace']:
        (tmp_path / folder / 'obsolete.py').write_bytes(b'')
    frozen['baseline_sha256'] = tree_hashes(tmp_path / 'baseline')
    frozen['verification_config_sha256'] = digest(tmp_path / 'verification-config.json')
    atomic_json(tmp_path / 'frozen.json', frozen)
    steps = [('bash', {'command': 'rm obsolete.py'}),
             ('write_file', {'file_path': 'package/__init__.py', 'content': ''}),
             ('write_file', {'file_path': 'package/calc.py', 'content': 'def square(x):\n    return x * x\n'}),
             ('bash', {'command': public['command']}), ('bash', {'command': SUBMIT})]
    with local_provider(lambda data, n: (200, response([steps[n - 1]]))) as (url, requests):
        agent = make_agent(tmp_path, monkeypatch, url, extensions=['trajectory_read'])
        assert agent.run('Create a package with square() and verify both signs.')['exit_status'] == 'Submitted'
        assert len(requests) == 5
    exported = export_artifact(tmp_path, {'reason': 'public new-project fixture'})
    patch = (tmp_path / 'model.patch').read_text(encoding='utf-8')
    assert 'package/__init__.py' in patch and '.pytest_cache' not in patch
    assert 'deleted file mode' in patch and 'obsolete.py' in patch
    assert exported['patch_bytes'] > 0 and verify_artifact(tmp_path)['autonomous_complete']
