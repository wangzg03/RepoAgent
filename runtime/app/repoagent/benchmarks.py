"""Benchmark adapters keep hidden tests and reference solutions out of inference."""
import gzip
import json
import shutil
import subprocess
import uuid
from pathlib import Path

from corecoder_mini.delivery import bind_verdict, digest, read_json
from corecoder_mini.engine import ScopedDocker
from corecoder_mini.task_state import atomic_json

ROOT = Path(__file__).resolve().parents[3]
OFFICIAL = ROOT / 'third_party/human_eval'
DEFAULT_SCORER_IMAGE = 'repoagent-scorer:local'


def verify_upstream():
    manifest = read_json(OFFICIAL / 'UPSTREAM.json')
    for name, expected in manifest['files'].items():
        path = OFFICIAL / name
        if not path.is_file():
            raise FileNotFoundError(
                'Optional official HumanEval assets missing. Run '
                'python scripts/prepare_humaneval.py --data-file <HumanEval.jsonl.gz> '
                'or explicitly use --download; no data is bundled or fetched automatically.')
        if digest(path) != expected:
            raise ValueError('official_humaneval_source_drift')
    return manifest


def problems():
    verify_upstream()
    with gzip.open(OFFICIAL / 'data/HumanEval.jsonl.gz', 'rt', encoding='utf-8') as stream:
        return {row['task_id']: row for row in (json.loads(line) for line in stream)}


def prepare_humaneval(task_id, packet, destination, *, profile=None, image=None, dollars=.10, seconds=600,
                      scorer_image=DEFAULT_SCORER_IMAGE):
    from .cli import DEFAULT_IMAGE, prepare
    problem = problems()[task_id]
    packet = Path(packet).resolve()
    # Only the public prompt is materialized in the inference input.
    seed = packet.parent / (packet.name + '-public-input')
    seed.mkdir(exist_ok=False, parents=True)
    (seed / 'solution.py').write_bytes((problem['prompt'] + '    raise NotImplementedError\n').encode('utf-8'))
    task = ('Implement the function in solution.py. Read it first; preserve the complete original prompt, '
            'imports, signature and docstring exactly. Replace only the unimplemented body, append helpers '
            'if necessary. Use standard-library Python. Run public examples and meaningful assertions, then submit.')
    config = prepare(task, destination, packet, seed, profile=profile, image=image or DEFAULT_IMAGE,
                     dollars=dollars, seconds=seconds, kind='implement',
                     verification={'kind': 'humaneval_official', 'task_id': task_id, 'seconds': 180,
                                   'upstream': verify_upstream(), 'scorer_image': scorer_image})
    config['task_id'] = task_id
    config['settings'].update(task_kind='benchmark', retain_experience=False)
    atomic_json(packet / 'task.json', config)
    return config


def prepare_swebench(public_instance, workspace, assets, linux_python, packet, destination, **kwargs):
    from .cli import prepare
    # Callers supply the original public baseline. Never inherit an old model patch.
    task_id, issue = public_instance['instance_id'], public_instance['problem_statement']
    config = prepare(issue, destination, packet, workspace, kind='fix', verification={
        'kind': 'swebench_official', 'assets_directory': str(Path(assets).resolve()),
        'baseline_source': str(Path(workspace).resolve()), 'linux_python': linux_python,
        'score_seconds': 1800}, **kwargs)
    config['task_id'] = task_id
    config['settings'].update(task_kind='benchmark', retain_experience=False)
    atomic_json(Path(packet) / 'task.json', config)
    return config


SCORER = r'''
import gzip,json,os,sys
from pathlib import Path
sys.path.insert(0,'/official')
from human_eval.evaluation import evaluate_functional_correctness
data=json.load(sys.stdin)
base=Path('/tmp/repoagent-official');base.mkdir()
problem=base/'problems.jsonl.gz'
with gzip.open(problem,'wt',encoding='utf-8') as f:
 f.write(json.dumps(data['problem'])+'\n')
sample=base/'samples.jsonl'
sample.write_text(json.dumps(data['sample'])+'\n',encoding='utf-8')
metrics=evaluate_functional_correctness(str(sample),k=[1],n_workers=1,timeout=3,problem_file=str(problem))
rows=[json.loads(line) for line in Path(str(sample)+'_results.jsonl').read_text().splitlines()]
print('REPOAGENT_OFFICIAL='+json.dumps({'metrics':metrics,'passed':rows[0]['passed'],'result':rows[0]['result']}))
'''


def run_scorer(argv, payload, seconds):
    name = 'repoagent-scorer-' + uuid.uuid4().hex
    argv = [*argv[:2], '--name', name, *argv[2:]]
    try:
        return subprocess.run(argv, input=json.dumps(payload), capture_output=True,
                              text=True, encoding='utf-8', timeout=seconds, check=False)
    finally:
        # A timed-out docker CLI alone does not stop its container.
        subprocess.run(['docker', 'rm', '-f', name], capture_output=True, timeout=15, check=False)


def verify_humaneval(directory, config):
    directory = Path(directory)
    manifest = verify_upstream()
    if config['upstream'] != manifest:
        raise ValueError('frozen_official_source_changed')
    frozen, result = read_json(directory / 'frozen.json'), read_json(directory / 'result.json')
    target = directory / ('verification-workspace-' + uuid.uuid4().hex)
    shutil.copytree(directory / 'baseline', target)
    sandbox = ScopedDocker(target, frozen['image_id'], frozen['job_id'])
    patch = (directory / 'model.patch').read_bytes().decode('utf-8')
    applied = sandbox.run('git apply -', target, 30, input_text=patch) if patch else None
    ok_apply = applied is None or applied.returncode == 0
    problem = problems()[config['task_id']]
    text = (target / 'solution.py').read_text(encoding='utf-8') if ok_apply else ''
    prefix_ok = text.startswith(problem['prompt'])
    report = {'passed': False, 'result': 'original_prompt_changed_or_patch_failed', 'metrics': None}
    if ok_apply and prefix_ok:
        payload = {'problem': problem, 'sample': {'task_id': config['task_id'], 'completion': text[len(problem['prompt']):]}}
        proc = run_scorer(['docker', 'run', '--rm', '-i', '--pull=never', '--network=none', '--read-only',
            '--label', 'corecoder.eval.job=' + frozen['job_id'], '--cap-drop=ALL', '--security-opt=no-new-privileges',
            '--user', '65534:65534', '--pids-limit=64', '--memory=512m', '--cpus=1',
            '--tmpfs', '/tmp:rw,exec,nosuid,nodev,size=64m,mode=1777',
            '--mount', 'type=bind,source=' + str(OFFICIAL.resolve()) + ',target=/official,readonly',
            '--entrypoint', 'python', config.get('scorer_image', DEFAULT_SCORER_IMAGE), '-B', '-c', SCORER],
            payload, config.get('seconds', 180))
        rows = [line.removeprefix('REPOAGENT_OFFICIAL=') for line in proc.stdout.splitlines() if line.startswith('REPOAGENT_OFFICIAL=')]
        if proc.returncode or len(rows) != 1:
            raise RuntimeError('official_scorer_failed; not scored as an agent failure')
        report = json.loads(rows[0])
    atomic_json(directory / 'official-report.json', {'task_id': config['task_id'], **report})
    evidence = directory / 'verification.json'
    verdict = {'ok': True, 'resolved': report['passed'], 'patch_applied': ok_apply,
               'patch_sha256': result['patch_sha256'], 'upstream_commit': manifest.get('commit'),
               'official_report_sha256': digest(directory / 'official-report.json'),
               'prompt_preserved': prefix_ok, 'hidden_tests_exposed_to_agent': False}
    if applied is not None and not ok_apply:
        verdict['patch_error'] = (applied.stdout + applied.stderr)[-2000:]
    atomic_json(evidence, verdict)
    return bind_verdict(directory, verdict, evidence, kind='humaneval_official')


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='benchmark', required=True)
    human = sub.add_parser('humaneval')
    human.add_argument('--ids', nargs='+', required=True, help='Explicit task IDs; no automatic retries')
    human.add_argument('--scorer-image', default=DEFAULT_SCORER_IMAGE)
    swe = sub.add_parser('swebench')
    swe.add_argument('--public-instance', type=Path, required=True)
    swe.add_argument('--workspace', type=Path, required=True)
    swe.add_argument('--assets', type=Path, required=True)
    swe.add_argument('--linux-python', required=True)
    for command in [human, swe]:
        command.add_argument('--output', type=Path, required=True)
        command.add_argument('--provider-config', type=Path)
        command.add_argument('--image')
        command.add_argument('--usd', type=float, default=.10)
        command.add_argument('--seconds', type=int, default=600)
        command.add_argument('--run', action='store_true')
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError('new_output_directory_required')
    args.output.mkdir(parents=True)
    from .cli import DEFAULT_IMAGE, execute
    prepared = []
    ids = args.ids if args.benchmark == 'humaneval' else [read_json(args.public_instance)['instance_id']]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate_benchmark_ids')
    for identity in ids:
        packet = args.output / identity.replace('/', '-')
        kwargs = {'profile': args.provider_config, 'image': args.image or DEFAULT_IMAGE,
                  'dollars': args.usd, 'seconds': args.seconds}
        if args.benchmark == 'humaneval':
            prepare_humaneval(identity, packet, args.output / (packet.name + '-delivery'),
                              scorer_image=args.scorer_image, **kwargs)
        else:
            prepare_swebench(read_json(args.public_instance), args.workspace, args.assets, args.linux_python,
                             packet, args.output / (packet.name + '-delivery'), **kwargs)
        prepared.append(packet)
    atomic_json(args.output / 'manifest.json', {'ids': ids, 'one_sample_per_task': True,
                                               'paid_tasks_started': bool(args.run), 'per_task_usd_cap': args.usd})
    if args.run:
        from .metrics import aggregate
        for packet in prepared:
            code = execute(packet)
            atomic_json(args.output / 'results.json', aggregate(prepared))
            if code:
                print('Stopped at first incomplete task; remaining tasks were not started.')
                return code
    print('Prepared benchmark tasks: ' + str(len(prepared)))
    return 0
