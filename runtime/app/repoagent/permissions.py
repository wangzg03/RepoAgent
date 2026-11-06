"""Operator policy is configuration, never model-produced intent."""
from .contracts import in_scope, safe_relative


def validate_settings(settings):
    scopes = settings.setdefault('write_scope', ['.'])
    if not isinstance(scopes, list) or not scopes:
        raise ValueError('write_scope_required')
    for scope in scopes:
        safe_relative(scope)
    caps = settings.setdefault('child_capabilities', ['read', 'test'])
    if not isinstance(caps, list) or set(caps) - {'read', 'test', 'write'} or 'read' not in caps:
        raise ValueError('invalid_child_capabilities')
    if not isinstance(settings.setdefault('mcp_servers', {}), dict):
        raise TypeError('mcp_servers_must_be_operator_configuration')
    if type(settings.setdefault('require_task_contract', True)) is not bool:
        raise ValueError('require_task_contract_boolean_required')


def authorize(environment, name, arguments):
    settings = environment.repoagent_settings
    if environment.depth:
        capabilities = settings.get('active_child_capabilities', ['read', 'test'])
        if name in {'edit_file', 'write_file'} and 'write' not in capabilities:
            raise ValueError('child_write_not_granted')
        if name in {'bash', 'run_tests'} and 'test' not in capabilities:
            raise ValueError('child_execution_not_granted; use read_file or code_search')
    if name in {'edit_file', 'write_file'}:
        path = safe_relative(arguments['file_path'])
        if not in_scope(path, settings.get('write_scope', ['.'])):
            raise ValueError('file_write_outside_operator_scope')
    if name == 'bash' and settings.get('write_scope', ['.']) != ['.']:
        from corecoder_mini.engine import SUBMIT
        command = arguments['command'].strip()
        # An arbitrary shell cannot enforce a path allowlist. Use file tools for
        # writes; configured test commands run in a throwaway copy (see run_tests).
        if command != SUBMIT and command != environment.public_check_command:
            raise ValueError('scoped_tasks_use_code_search_read_file_and_run_tests; arbitrary_shell_disabled')
    if (name.startswith('mcp__') and settings.get('write_scope', ['.']) != ['.']
            and not environment.registry.tools[name].read_only):
        raise ValueError('mutating_mcp_disabled_for_scoped_tasks')
