"""New capabilities attach through the existing registry and action journal."""
import re

from corecoder_mini.task_state import atomic_json

from .contracts import install
from .localization import CodeIndex, CodeSearchTool


def extend(state, environment, settings, workspace, directory, depth):
    environment.repoagent_settings = settings
    environment.registry.register(CodeSearchTool(CodeIndex(workspace, directory / 'symbol-index.json')))
    if not depth and settings.get('require_task_contract', True):
        install(state, environment, settings)
    from .verification import RunTestsTool
    environment.registry.register(RunTestsTool(environment, directory))
    if not depth:
        from .mcp import MCPClient
        environment.mcp_clients = []
        sandbox = environment.registry.tools['bash'].sandbox
        for name, spec in settings.get('mcp_servers', {}).items():
            if not re.fullmatch(r'[A-Za-z0-9_-]{1,40}', name) or not isinstance(spec, dict):
                raise ValueError('invalid_mcp_server')
            command, args = spec.get('command'), spec.get('args', [])
            if not isinstance(command, str) or not command or not isinstance(args, list) or not all(isinstance(a, str) for a in args):
                raise ValueError('invalid_mcp_command')
            client = MCPClient(name, command, args, sandbox=sandbox)
            environment.mcp_clients.append(client)
            for tool in client.tools:
                if tool._remote_name in spec.get('trusted_read_only_tools', []):
                    tool.read_only = tool.concurrency_safe = True
                environment.registry.register(tool)
    atomic_json(directory / 'tool-catalog.json', catalog(environment.registry))


def catalog(registry):
    return [{**item, 'execution': {'read_only': registry.tools[item['name']].read_only,
                                 'concurrency_safe': registry.tools[item['name']].concurrency_safe,
                                 'journaled': True, 'cancellation': 'task_deadline',
                                 'output': 'bounded; captured receipts are recoverable; transport caps may truncate'}}
            for item in registry.catalog()]
