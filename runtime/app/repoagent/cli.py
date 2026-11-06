"""Interactive demo and benchmark preparation share one execution pipeline."""
import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

from corecoder_mini.delivery import read_json
from corecoder_mini.task_state import atomic_json

ROOT = Path(__file__).resolve().parents[3]
RUNTIME = ROOT / 'runtime'
DEFAULT_IMAGE = 'repoagent-sandbox:local'


def provider(path=None):
    config = read_json(Path(path) if path else RUNTIME / 'API_PROVIDER_CURRENT.json')
    if not config:
        raise ValueError('provider_configuration_missing')
    return config


def prepare(task, destination, packet, source=None, *, profile=None, image=DEFAULT_IMAGE,
            dollars=.10, seconds=600, check=None, write_scope=None, child_write=False, kind=None,
            verification=None, mcp_servers=None, tool_extensions=None, memory_embedding='off'):
    from .permissions import validate_settings
    packet, destination = Path(packet).resolve(), Path(destination).resolve()
    if not task.strip() or len(task) > 50000 or destination.exists() or packet.exists():
        raise ValueError('provide_task_and_new_packet_and_destination')
    if destination == ROOT or ROOT.is_relative_to(destination) or destination.is_relative_to(RUNTIME):
        raise ValueError('destination_overlaps_runtime')
    if packet.is_relative_to(destination) or destination.is_relative_to(packet):
        raise ValueError('run_records_and_delivery_must_be_separate')
    if dollars <= 0 or seconds <= 0:
        raise ValueError('positive_budget_and_deadline_required')
    source = Path(source).resolve(strict=True) if source else None
    if source and (packet.is_relative_to(source) or destination.is_relative_to(source) or source.is_relative_to(destination)):
        raise ValueError('source_run_and_delivery_overlap')
    selected = provider(profile)
    project_scope = str(source or destination)
    settings = {'repoagent': True, 'require_task_contract': True, 'task_kind': 'user', 'retain_experience': False,
                'step_limit': 40, 'request_limit': 40, 'tool_limit': 80, 'token_limit': 600000,
                'dollar_limit': dollars, 'agent_seconds': seconds, 'request_seconds': 180, 'tool_seconds': 120,
                'context_tokens': selected.get('context_tokens', 32000), 'max_output_tokens': 8192,
                'provider_stream': True, 'history_target_tokens': 8000, 'history_output_chars': 2000,
                'keep_recent_turns': 4, 'max_subagents': 2, 'child_concurrency': 2, 'child_requests': 5,
                'write_scope': write_scope or ['.'],
                'child_capabilities': ['read', 'test', 'write'] if child_write else ['read', 'test'],
                'operator_workspace': {'source': str(source) if source else None, 'destination': str(destination),
                                       'execution_root': '/workspace', 'mode': 'existing' if source else 'new'},
                'memory_embedding': memory_embedding, 'contract_verification': check is None,
                'project_scope': project_scope,
                'memory_path': str(ROOT / 'memory' / (hashlib.sha256(project_scope.casefold().encode()).hexdigest()[:24] + '.sqlite3')),
                'mcp_servers': mcp_servers or {}, 'tool_extensions': tool_extensions or [],
                'reasoning_effort': selected.get('reasoning_effort', 'high')}
    if kind:
        settings['task_intent'] = kind
    if check:
        settings['public_check_command'] = check
    validate_settings(settings)
    packet.mkdir(parents=True)
    seed = source or packet / 'seed'
    if not source:
        seed.mkdir()
    config = {'task_id': 'repoagent-' + uuid.uuid4().hex[:12], 'task': task.strip(), 'workspace': str(seed),
              'provider': selected.get('provider', 'openai'), 'model': selected['default_model'],
              'base_url': selected['base_url'], 'api_key_env': selected.get('api_key_env', 'DEEPSEEK_API_KEY'),
              'pricing_usd_per_million': selected.get('pricing_usd_per_million', [.30, 1.20, .006]),
              'image': image, 'settings': settings,
              'verification': verification or ({'kind': 'local_command', 'command': check, 'seconds': 180}
                                                if check else {'kind': 'repoagent_contract'})}
    atomic_json(packet / 'task.json', config)
    atomic_json(packet / 'demo.json', {'destination': str(destination), 'harness': str(RUNTIME),
                                      'provider_profile': str(Path(profile).resolve()) if profile else str(RUNTIME/'API_PROVIDER_CURRENT.json'),
                                      'model_calls_during_preparation': 0})
    return config


def execute(packet):
    from . import legacy_launcher
    from .metrics import summarize
    packet = Path(packet).resolve()
    code = legacy_launcher.run_packet(packet)
    if (packet / 'run/agent/task-contract.json').is_file():
        shutil.copyfile(packet / 'run/agent/task-contract.json', packet / 'task-contract.json')
    if (packet / 'run/result.json').is_file():
        metrics = summarize(packet / 'run')
        print('Observed metrics: ' + str(packet / 'metrics.json'))
        print(json.dumps({k: metrics[k] for k in ['state', 'model_submitted', 'artifact_correct', 'autonomous_complete']}))
    return code


def export_tools(output):
    from types import SimpleNamespace

    from corecoder_mini.budget import Budget
    from corecoder_mini.engine import build_agent
    from corecoder_mini.supervisor import DEFAULTS

    from .integration import catalog
    with tempfile.TemporaryDirectory() as temp:
        base = Path(temp)
        (base / 'workspace').mkdir()
        settings = {**DEFAULTS, 'repoagent': True, 'memory_embedding': 'off'}
        agent = build_agent(SimpleNamespace(budget=Budget(.10)), base / 'workspace', base / 'agent', DEFAULT_IMAGE, settings)
        rows = catalog(agent.env.registry)
        atomic_json(Path(output), rows)
        agent.env.cleanup()
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description='RepoAgent: natural language -> task JSON -> sandbox coding -> verified delivery')
    parser.add_argument('--task')
    parser.add_argument('--directory', type=Path, help='New delivery directory; existing source is never overwritten')
    parser.add_argument('--workspace', type=Path, help='Optional existing repository input')
    parser.add_argument('--provider-config', type=Path)
    parser.add_argument('--image', default=DEFAULT_IMAGE)
    parser.add_argument('--usd', type=float, default=.10)
    parser.add_argument('--seconds', type=int, default=600)
    parser.add_argument('--check', help='Operator-fixed public check; model cannot replace it')
    parser.add_argument('--kind', choices=['implement', 'fix', 'refactor', 'test', 'explain', 'locate', 'review'])
    parser.add_argument('--write-scope', nargs='+', default=['.'])
    parser.add_argument('--allow-child-write', action='store_true', help='Allow model to request scoped isolated child edits')
    parser.add_argument('--mcp-config', type=Path, help='Operator-owned JSON with mcpServers; commands run inside Docker')
    parser.add_argument('--tool-extension', action='append', default=[], help='Allowlisted installed corecoder.tools entry point')
    parser.add_argument('--memory-embedding', choices=['off', 'ollama'], default='off')
    parser.add_argument('--packet', type=Path)
    parser.add_argument('--run', action='store_true', help='Run model and Docker; API provider incurs cost')
    parser.add_argument('--run-packet', type=Path)
    parser.add_argument('--export-tools', type=Path)
    parser.add_argument('--export-schema', type=Path)
    parser.add_argument('--metrics', type=Path, help='Existing run directory; read-only measurement')
    args = parser.parse_args(argv)
    try:
        if args.export_schema:
            from .contracts import Intent
            atomic_json(args.export_schema, Intent.model_json_schema())
            return 0
        if args.export_tools:
            rows = export_tools(args.export_tools)
            print('Registered tools: ' + str(len(rows)))
            return 0
        if args.metrics:
            from .metrics import summarize
            print(json.dumps(summarize(args.metrics), ensure_ascii=False, indent=2))
            return 0
        if args.run_packet:
            return execute(args.run_packet)
        task = args.task or input('Task / 描述任务: ')
        destination = args.directory or Path(input('Delivery directory / 新交付目录: '))
        packet = (args.packet or ROOT / 'runs' / uuid.uuid4().hex[:12]).resolve()
        prepare(task, destination, packet, args.workspace, profile=args.provider_config, image=args.image,
                dollars=args.usd, seconds=args.seconds, check=args.check, write_scope=args.write_scope,
                child_write=args.allow_child_write, kind=args.kind,
                mcp_servers=read_json(args.mcp_config)['mcpServers'] if args.mcp_config else None,
                tool_extensions=args.tool_extension, memory_embedding=args.memory_embedding)
        print('Prepared operator configuration: ' + str(packet / 'task.json'))
        print('Model intent JSON after the first task_contract call: ' + str(packet / 'run/agent/task-contract.json'))
        print('Preparation does not call a model. Metrics proposed by a model are not measured results.')
        return execute(packet) if args.run else 0
    except Exception as exc:  # noqa: BLE001 - no provider exception bodies or secrets on console
        print('Stopped: ' + (str(exc) if isinstance(exc, ValueError) else type(exc).__name__), file=sys.stderr)
        return 1
