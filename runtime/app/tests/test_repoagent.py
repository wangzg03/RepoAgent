"""New integration behavior; no paid inference or benchmark answers in prompts."""
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from corecoder_mini.budget import Budget
from corecoder_mini.delivery import read_json
from corecoder_mini.engine import SUBMIT, build_agent
from corecoder_mini.runtime import local
from corecoder_mini.supervisor import DEFAULTS, validate_extensions
from corecoder_mini.task_state import atomic_json
from repoagent.child_permissions import assignment, integrate, validate_changes
from repoagent.cli import prepare
from repoagent.localization import CodeIndex
from repoagent.metrics import localization


@pytest.fixture(autouse=True)
def clear_context():
    vars(local).clear()
    yield
    vars(local).clear()


def contract():
    return {'kind': 'implement', 'summary': 'Implement addition',
            'requirements': [{'id': 'R1', 'behavior': 'add two numbers'}],
            'metrics': [{'id': 'M1', 'name': 'behavior', 'kind': 'behavior', 'target': 'all public assertions pass',
                         'requirement_ids': ['R1'], 'check_ids': ['C1']}],
            'checks': [{'id': 'C1', 'command': 'python -m unittest discover -s tests -v'}],
            'plan': [{'id': 'T1', 'content': 'Implement and test', 'requirements': ['R1'],
                      'write_scope': ['src'], 'evidence': 'C1 passes'}]}


def agent(tmp_path, **extra):
    workspace = tmp_path / 'workspace'
    workspace.mkdir(exist_ok=True)
    settings = {**DEFAULTS, 'repoagent': True, 'memory_embedding': 'off', **extra}
    validate_extensions(settings)
    result = build_agent(SimpleNamespace(budget=Budget(1)), workspace, tmp_path / 'agent', 'fixture-image', settings)
    result.working_state.bind_goal('Implement addition without dependencies')
    return result


def test_contract_registered_bounded_and_writes_real_json(tmp_path):
    coding = agent(tmp_path, write_scope=['src'])
    registry = coding.env.registry
    assert {'task_contract', 'code_search', 'read_file', 'edit_file', 'write_file', 'run_tests',
            'scratchpad', 'todo_write', 'memory', 'trajectory_read', 'subagent', 'bash'} == set(registry.tools)
    data = contract()
    registry.validate('task_contract', data)
    observed = coding.env.execute({'tool': 'task_contract', 'arguments': data})
    assert observed['returncode'] == 0
    record = read_json(tmp_path / 'agent/task-contract.json')
    assert record['allowed_write_scope'] == ['src']
    assert record['original_request_sha256']
    assert record['metrics_are_proposed']
    assert record['intent']['metrics'][0]['source'] == 'inferred'
    assert coding.working_state.data['todos'][0]['status'] == 'pending'
    assert coding.env.public_check_command


@pytest.mark.parametrize('mutation', ['unknown_reference', 'cycle', 'scope', 'missing_checks', 'authority'])
def test_contract_rejects_invalid_or_expanded_permissions(tmp_path, mutation):
    coding = agent(tmp_path, write_scope=['src'])
    data = contract()
    if mutation == 'unknown_reference':
        data['metrics'][0]['requirement_ids'] = ['R2']
    elif mutation == 'cycle':
        data['plan'][0]['dependencies'] = ['T1']
    elif mutation == 'scope':
        data['plan'][0]['write_scope'] = ['other']
    elif mutation == 'missing_checks':
        data['checks'] = []
    else:
        data['workspace'] = {'root': 'C:/'}
    with pytest.raises(ValueError):
        coding.env.registry.tools['task_contract'].execute(**data)
    assert not (tmp_path / 'agent/task-contract.json').exists()


def test_contract_cannot_change_frozen_metrics_or_operator_check(tmp_path):
    coding = agent(tmp_path, public_check_command='python -m unittest -v')
    tool = coding.env.registry.tools['task_contract']
    tool.execute(**contract())
    data = contract()
    data['metrics'][0]['target'] = 'ignore failures'
    with pytest.raises(ValueError, match='frozen'):
        tool.execute(**data)
    assert coding.env.public_check_command == 'python -m unittest -v'


def test_writes_before_contract_or_outside_scope_do_not_execute(tmp_path, monkeypatch):
    coding = agent(tmp_path, write_scope=['src'])
    called = []
    monkeypatch.setattr(coding.env.registry.tools['write_file'], 'execute', lambda **kwargs: called.append(kwargs))
    with pytest.raises(ValueError, match='task_contract'):
        coding.env.execute({'tool': 'write_file', 'arguments': {'file_path': 'src/a.py', 'content': 'x=1'}})
    coding.env.registry.tools['task_contract'].execute(**contract())
    with pytest.raises(ValueError, match='scope'):
        coding.env.execute({'tool': 'write_file', 'arguments': {'file_path': 'other/a.py', 'content': 'x=1'}})
    assert not called


def test_ast_index_ranking_references_incremental_rename_and_deletion(tmp_path):
    root, cache = tmp_path / 'repo', tmp_path / 'state/index.json'
    root.mkdir()
    (root / 'target.py').write_text('def target(x):\n    return x\n\ndef caller():\n    return target(1)\n', encoding='utf-8')
    (root / 'noise.py').write_text('# target appears in a comment\n', encoding='utf-8')
    (root / '.env.py').write_text('def target(): pass\n', encoding='utf-8')
    index = CodeIndex(root, cache)
    first = index.search('target', 'definition')
    assert first['hits'][0]['path'] == 'target.py'
    assert first['hits'][0]['symbol'] == 'target'
    assert first['index']['reparsed'] == 2
    assert index.search('target')['index']['reparsed'] == 0
    assert any('target(1)' in hit['text'] for hit in index.search('target', 'references')['hits'])
    call = index.search('target', 'calls')['hits'][0]
    assert call['relation'] == 'syntactic_call' and call['symbol'] == 'caller'
    (root / 'target.py').write_text('def renamed():\n    return 1\n', encoding='utf-8')
    assert not index.search('target', 'definition')['hits']
    assert index.search('renamed', 'definition')['hits']
    (root / 'target.py').unlink()
    assert index.search('renamed')['index']['removed'] == 1
    assert not CodeIndex(root, cache).search('renamed')['hits']


def test_partial_python_still_supports_literal_search_and_scope(tmp_path):
    (tmp_path / 'broken.py').write_text('def broken(:\n    important_name\n', encoding='utf-8')
    index = CodeIndex(tmp_path, tmp_path.parent / 'index.json')
    result = index.search('important_name')
    assert result['hits'] and index.records['broken.py']['parse_error']
    with pytest.raises(ValueError):
        index.search('important_name', path_prefix='../')


def test_child_permissions_are_operator_subset_not_model_authority():
    settings = {'child_capabilities': ['read', 'test'], 'write_scope': ['src']}
    with pytest.raises(ValueError, match='authorized'):
        assignment({'capabilities': ['read', 'write'], 'write_scope': ['src']}, settings)
    settings['child_capabilities'].append('write')
    assert assignment({'capabilities': ['read', 'write'], 'write_scope': ['src/a.py']}, settings)['write_scope'] == ['src/a.py']
    with pytest.raises(ValueError, match='parent'):
        assignment({'capabilities': ['read', 'write'], 'write_scope': ['tests']}, settings)


def test_child_integration_conflicts_are_rejected_before_any_parent_write(tmp_path):
    from scripts.swebench_io import tree_hashes
    parent = tmp_path / 'parent'
    child = tmp_path / 'agent/children/c1/workspace'
    parent.mkdir()
    child.mkdir(parents=True)
    (parent / 'a.py').write_text('old\n', encoding='utf-8')
    baseline = tree_hashes(parent)
    (child / 'a.py').write_text('new\n', encoding='utf-8')
    record = {'status': 'completed', 'input': {'capabilities': ['read', 'write'], 'write_scope': ['a.py']},
              'changed_paths': ['a.py'], 'baseline_sha256': baseline, 'after_sha256': tree_hashes(child),
              'review': {'disposition': 'used'}}
    manager = SimpleNamespace(lock=threading.RLock(), records={'c1': record}, workspace=parent,
                              directory=tmp_path/'agent', save=lambda: None)
    (parent / 'a.py').write_text('parent changed\n', encoding='utf-8')
    with pytest.raises(ValueError, match='conflict'):
        integrate(manager, 'c1')
    assert (parent/'a.py').read_text() == 'parent changed\n'
    (parent/'a.py').write_text('old\n', encoding='utf-8')
    assert integrate(manager, 'c1')['main_verification_required']
    assert (parent/'a.py').read_text() == 'new\n'
    with pytest.raises(ValueError, match='already_attempted'):
        integrate(manager, 'c1')


def test_localization_metrics_distinguish_ranking_and_missing_targets(tmp_path):
    (tmp_path/'a.py').write_text('def alpha(): pass\n', encoding='utf-8')
    index = CodeIndex(tmp_path, tmp_path.parent/'index.json')
    result = localization(index, [{'id':'hit','query':'alpha','expected_paths':['a.py']},
                                  {'id':'miss','query':'missing','expected_paths':['a.py']}], k=5)
    assert result['aggregation']['recall_at_k'] == .5
    assert result['aggregation']['mrr'] == .5
    assert result['aggregation']['precision_at_k'] == .1


def test_batch_metrics_keep_unstarted_and_unscored_cases_in_denominators(tmp_path):
    from repoagent.metrics import aggregate
    one,two=tmp_path/'one',tmp_path/'two'
    atomic_json(one/'run/result.json',{'state':'completed','official_correct':True,'autonomous_complete':True})
    summary=aggregate([one,two])
    assert summary['planned']==2 and summary['started']==1
    assert summary['autonomous_completion_rate']==.5
    assert summary['official_correct_rate'] is None
    assert not summary['official_results_complete']


def test_prepare_is_offline_and_existing_source_uses_same_pipeline(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    source.mkdir()
    config = prepare('Fix existing behavior', tmp_path/'delivery', tmp_path/'packet', source,
                     dollars=.1, seconds=600, check='python -m unittest -v', kind='fix')
    assert config['settings']['operator_workspace']['mode'] == 'existing'
    assert config['settings']['task_intent'] == 'fix'
    assert config['verification']['kind'] == 'local_command'
    assert not (tmp_path/'delivery').exists()
    followup=prepare('Continue fixing',tmp_path/'delivery2',tmp_path/'packet2',source)
    assert config['settings']['memory_path']==followup['settings']['memory_path']
    assert config['settings']['project_scope']==str(source.resolve())


@pytest.mark.skipif(not (Path(__file__).resolve().parents[3] /
    'third_party/human_eval/data/HumanEval.jsonl.gz').is_file(),
    reason='optional official HumanEval data is not bundled')
def test_humaneval_prepare_exposes_only_public_prompt(tmp_path):
    from repoagent.benchmarks import prepare_humaneval, problems
    problem = problems()['HumanEval/0']
    config = prepare_humaneval('HumanEval/0', tmp_path/'packet', tmp_path/'delivery')
    public = Path(config['workspace']) / 'solution.py'
    assert public.read_text(encoding='utf-8') == problem['prompt'] + '    raise NotImplementedError\n'
    assert problem['test'] not in json.dumps(config)
    assert config['settings']['task_kind'] == 'benchmark'
    assert not (tmp_path/'packet/run').exists()


def test_swebench_prepare_uses_public_issue_and_keeps_official_adapter(tmp_path):
    from repoagent.benchmarks import prepare_swebench
    source=tmp_path/'original-baseline'
    source.mkdir()
    instance={'instance_id':'example__repo-1','problem_statement':'Public bug description',
              'patch':'hidden-reference-must-not-enter-prompt','test_patch':'hidden-tests-must-not-enter-prompt'}
    config=prepare_swebench(instance,source,tmp_path/'assets','/existing/wsl/python',
                            tmp_path/'packet',tmp_path/'delivery')
    assert config['task']=='Public bug description'
    assert 'hidden-reference' not in json.dumps(config)
    assert config['verification']['kind']=='swebench_official'
    assert config['settings']['task_intent']=='fix'
    assert config['settings']['task_kind']=='benchmark'


def test_scoped_child_detects_outside_mutation():
    with pytest.raises(ValueError, match='scope'):
        validate_changes({'input': {'capabilities':['read','write'], 'write_scope':['src']}}, ['tests/a.py'])


def test_native_protocol_transports_model_contract_and_records_without_api(tmp_path, monkeypatch):
    from test_native_upstream import local_provider, make_agent, response
    data = contract()
    def reply(payload, number):
        names = [t['function']['name'] for t in payload['tools']]
        assert 'task_contract' in names and 'code_search' in names
        return 200, response([('task_contract', data)])
    with local_provider(reply) as (url, requests):
        coding = make_agent(tmp_path, monkeypatch, url, extensions=['todo_write','scratchpad'], repoagent=True)
        coding.working_state.bind_goal('Implement addition')
        message = coding.model.query([{'role':'system','content':'Public test fixture'},
                                      {'role':'user','content':'Implement addition'}])
        action = message['extra']['actions'][0]
        result = coding.env.execute(action)
        assert result['returncode'] == 0
        assert len(requests) == 1
        assert coding.model.provider.budget.requests == 1
        assert read_json(tmp_path/'agent/task-contract.json')['intent']['summary'] == 'Implement addition'


def test_mcp_cannot_launch_host_command():
    from repoagent.mcp import MCPClient, MCPError
    with pytest.raises(MCPError, match='Docker'):
        MCPClient('blocked', 'python', ['-c','print(1)'])


def test_official_scorer_timeout_cleans_only_its_owned_container(monkeypatch):
    import subprocess

    from repoagent.benchmarks import run_scorer
    calls=[]
    def invoke(argv,**kwargs):
        calls.append(argv)
        if len(calls)==1:
            raise subprocess.TimeoutExpired(argv,1)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(subprocess,'run',invoke)
    with pytest.raises(subprocess.TimeoutExpired):
        run_scorer(['docker','run','--rm','fixture'],{},1)
    assert calls[1]==['docker','rm','-f',calls[0][3]]
    assert calls[0][3].startswith('repoagent-scorer-')


@pytest.mark.docker
def test_humaneval_adapter_rebuilds_patch_and_binds_actual_official_evidence(tmp_path):
    from corecoder_mini.delivery import digest, export_artifact
    from corecoder_mini.supervisor import verify_artifact
    from repoagent.benchmarks import DEFAULT_SCORER_IMAGE, problems, verify_upstream
    from repoagent.cli import DEFAULT_IMAGE
    from scripts.swebench_io import tree_hashes
    # A scorer-side positive control, never a generated-model result.
    problem=problems()['HumanEval/0']
    run=tmp_path/'run'
    (run/'baseline').mkdir(parents=True)
    (run/'workspace').mkdir()
    (run/'baseline/solution.py').write_text(problem['prompt']+'    raise NotImplementedError\n',encoding='utf-8')
    (run/'workspace/solution.py').write_text(problem['prompt']+problem['canonical_solution'],encoding='utf-8')
    verification={'kind':'humaneval_official','task_id':'HumanEval/0','upstream':verify_upstream(),
                  'scorer_image':DEFAULT_SCORER_IMAGE,'seconds':120}
    atomic_json(run/'verification-config.json',verification)
    atomic_json(run/'frozen.json',{'task_id':'HumanEval/0','settings':{},'image_id':DEFAULT_IMAGE,
        'job_id':'repoagent-official-fixture','baseline_sha256':tree_hashes(run/'baseline'),
        'verification_config_sha256':digest(run/'verification-config.json')})
    export_artifact(run,{'reason':'scorer_control','inference_started':False})
    result=verify_artifact(run)
    assert result['official_correct'] is True and result['artifact_correct'] is True
    assert not result['model_submitted'] and not result['autonomous_complete']
    assert read_json(run/'verification.json')['hidden_tests_exposed_to_agent'] is False


@pytest.mark.docker
def test_real_docker_mcp_handshake_tool_call_and_no_provider_key(tmp_path, monkeypatch):
    from corecoder_mini.native_docker import SessionDocker
    from repoagent.cli import DEFAULT_IMAGE
    from repoagent.mcp import MCPClient
    monkeypatch.setenv('DEEPSEEK_API_KEY','private-fixture-never-forward')
    workspace = tmp_path/'mcp-workspace'
    workspace.mkdir()
    server = '''import json,sys,os
for line in sys.stdin:
 m=json.loads(line)
 if 'id' not in m: continue
 method=m['method']
 if method=='initialize': r={'protocolVersion':'2025-06-18','capabilities':{'tools':{}},'serverInfo':{'name':'fixture','version':'1'}}
 elif method=='tools/list': r={'tools':[{'name':'echo','inputSchema':{'type':'object','properties':{'text':{'type':'string'}},'required':['text']}}]}
 elif method=='tools/call': r={'content':[{'type':'text','text':m['params']['arguments']['text']+'|credential='+str('DEEPSEEK_API_KEY' in os.environ)}]}
 else: r={}
 print(json.dumps({'jsonrpc':'2.0','id':m['id'],'result':r}),flush=True)
'''
    sandbox = SessionDocker(workspace, DEFAULT_IMAGE, 'repoagent-mcp-fixture')
    client = None
    try:
        client = MCPClient('fixture','python',['-u','-c',server],sandbox=sandbox)
        tool = next(t for t in client.tools if t.name == 'mcp__fixture__echo')
        assert not tool.read_only and not tool.concurrency_safe
        assert tool.execute(text='observed') == 'observed|credential=False'
    finally:
        if client:
            client.close()
        sandbox.cleanup()


@pytest.mark.docker
def test_scripted_model_child_edit_review_integration_and_main_retest(tmp_path):
    from test_harness import ScriptedProvider

    from repoagent.cli import DEFAULT_IMAGE
    workspace = tmp_path/'workspace'
    workspace.mkdir()
    (workspace/'main.py').write_text('def add(a,b):\n    return a-b\n', encoding='utf-8')
    intent = contract()
    intent['kind'] = 'fix'
    intent['plan'][0]['write_scope'] = ['main.py']
    intent['checks'][0]['command'] = "python -c 'from main import add; assert add(2,3)==5; assert add(-1,1)==0'"
    parent = ScriptedProvider([
        ('task_contract',intent),
        ('subagent',{'action':'spawn','jobs':[{'id':'patch','task':'Correct the arithmetic in main.py',
                     'capabilities':['read','write'],'write_scope':['main.py']}]}),
        ('subagent',{'action':'review','identity':'patch','disposition':'used','reason':'Actual edit receipt addresses R1'}),
        ('subagent',{'action':'integrate','identity':'patch'}),
        ('run_tests',{'command':intent['checks'][0]['command']}),
        ('bash',{'command':SUBMIT})])
    child = ScriptedProvider([
        ('edit_file',{'file_path':'main.py','old_string':'return a-b','new_string':'return a+b'}),
        ('subagent_report',{'summary':'Corrected addition in granted file',
                           'findings':[{'claim':'Changed subtraction to addition','action':1}]})])
    settings={**DEFAULTS,'repoagent':True,'memory_embedding':'off','child_capabilities':['read','test','write'],
              'child_requests':4,'tool_limit':24}
    coding=build_agent(parent,workspace,tmp_path/'agent',DEFAULT_IMAGE,settings,
                       child_provider_factory=lambda identity,budget: child)
    coding.working_state.bind_goal('Fix addition in main.py and verify')
    try:
        result=coding.run('Fix addition in main.py and verify')
        assert result['exit_status']=='Submitted'
        record=coding.children.records['patch']
        assert record['status']=='completed'
        assert record['changed_paths']==['main.py']
        assert record['integration']['status']=='integrated'
        assert 'subagent' not in {t['name'] for t in read_json(tmp_path/'agent/children/patch/agent/tool-catalog.json')}
        assert 'return a+b' in (workspace/'main.py').read_text()
        assert child.budget.shared is coding.children.gate.shared
        assert read_json(tmp_path/'agent/actions/0005.json')['verification_candidate']
    finally:
        coding.env.cleanup()


@pytest.mark.docker
def test_real_docker_contract_write_test_submission_and_independent_patch(tmp_path):
    from corecoder_mini.delivery import export_artifact
    from corecoder_mini.supervisor import verify_artifact
    from repoagent.cli import DEFAULT_IMAGE
    from scripts.swebench_io import tree_hashes
    workspace = tmp_path/'run/workspace'
    workspace.mkdir(parents=True)
    baseline = tmp_path/'run/baseline'
    baseline.mkdir()
    settings = {**DEFAULTS, 'repoagent':True, 'memory_embedding':'off', 'max_subagents':0}
    coding = build_agent(SimpleNamespace(budget=Budget(1)), workspace, tmp_path/'run/agent', DEFAULT_IMAGE, settings)
    coding.working_state.bind_goal('Create an addition module and assert behavior')
    data = contract()
    data['plan'][0]['write_scope'] = ['.']
    data['checks'][0]['command'] = "python -c 'from main import add; assert add(2,3)==5; assert add(-2,1)==-1'"
    try:
        coding.env.execute({'tool':'task_contract','arguments':data})
        coding.env.execute({'tool':'write_file','arguments':{'file_path':'main.py','content':'def add(a,b):\n    return a+b\n'}})
        observed = coding.env.execute({'tool':'run_tests','arguments':{'command':data['checks'][0]['command']}})
        assert observed['returncode'] == 0
        from corecoder_mini.upstream import Submitted
        with pytest.raises(Submitted):
            coding.env.execute({'tool':'bash','arguments':{'command':SUBMIT}})
    finally:
        coding.env.cleanup()
    run=tmp_path/'run'
    atomic_json(run/'frozen.json', {'task_id':'original-dev-addition', 'image_id':DEFAULT_IMAGE, 'job_id':'repoagent-dev',
                                   'baseline_sha256':tree_hashes(baseline), 'settings':settings})
    export_artifact(run, {'reason':'worker_exit','inference_started':False})
    atomic_json(run/'verification-config.json', {'kind':'repoagent_contract'})
    frozen=read_json(run/'frozen.json')
    from corecoder_mini.delivery import digest
    frozen['verification_config_sha256']=digest(run/'verification-config.json')
    atomic_json(run/'frozen.json', frozen)
    result=verify_artifact(run)
    assert result['autonomous_complete'] and result['artifact_correct']


@pytest.mark.docker
def test_real_docker_official_humaneval_positive_and_negative_controls(tmp_path):
    import subprocess

    from repoagent.benchmarks import DEFAULT_SCORER_IMAGE, OFFICIAL, SCORER, problems
    problem = problems()['HumanEval/0']
    for completion, expected in [(problem['canonical_solution'], True), ('\n    raise AssertionError()\n', False)]:
        observed = subprocess.run(['docker','run','--rm','-i','--pull=never','--network=none','--read-only',
            '--user','65534:65534','--cap-drop=ALL','--security-opt=no-new-privileges','--pids-limit=64',
            '--memory=512m','--cpus=1','--tmpfs','/tmp:rw,exec,size=64m,mode=1777',
            '--mount','type=bind,source='+str(OFFICIAL.resolve())+',target=/official,readonly',
            '--entrypoint','python',DEFAULT_SCORER_IMAGE,'-B','-c',SCORER],
            input=json.dumps({'problem':problem,'sample':{'task_id':'HumanEval/0','completion':completion}}),
            capture_output=True,text=True,timeout=90,check=False)
        assert observed.returncode == 0, observed.stderr[-1500:]
        row = next(line for line in observed.stdout.splitlines() if line.startswith('REPOAGENT_OFFICIAL='))
        assert json.loads(row.removeprefix('REPOAGENT_OFFICIAL='))['passed'] is expected
