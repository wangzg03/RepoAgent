"""Narrow child capabilities and explicitly integrate reviewed isolated changes."""
import uuid

from corecoder_mini.delivery import atomic_bytes
from corecoder_mini.task_state import atomic_json

from .contracts import in_scope, safe_relative


def assignment(job, settings):
    granted = job.get('capabilities', ['read', 'test'])
    allowed = settings.get('child_capabilities', ['read', 'test'])
    if not isinstance(granted, list) or 'read' not in granted or set(granted) - set(allowed):
        raise ValueError('child_capability_not_operator_authorized')
    scopes = job.get('write_scope', [])
    if 'write' in granted and not scopes:
        raise ValueError('child_write_requires_explicit_scope')
    if 'write' not in granted and scopes:
        raise ValueError('read_child_cannot_request_write_scope')
    for scope in scopes:
        safe_relative(scope)
        if not in_scope(scope, settings.get('write_scope', ['.'])):
            raise ValueError('child_scope_exceeds_parent')
    return {'capabilities': granted, 'write_scope': scopes}


def validate_changes(record, paths):
    capabilities = record['input'].get('capabilities', ['read', 'test'])
    if paths and 'write' not in capabilities:
        raise ValueError('read_only_child_snapshot_changed')
    for path in paths:
        safe_relative(path)
        if not in_scope(path, record['input']['write_scope']):
            raise ValueError('child_changed_outside_granted_scope')


def integrate(manager, identity):
    """No Git, no replay, preflight all conflicts; keep an intent for interrupted writes."""
    from scripts.swebench_io import tree_hashes
    with manager.lock:
        record = manager.records.get(identity, {})
        if record.get('status') != 'completed' or (record.get('review') or {}).get('disposition') != 'used':
            raise ValueError('completed_child_and_explicit_parent_review_required')
        if record.get('integration'):
            raise ValueError('child_integration_already_attempted; inspect_saved_intent')
        changes = record.get('changed_paths', [])
        validate_changes(record, changes)
        if not changes:
            raise ValueError('child_has_no_patch')
        child = manager.directory / 'children' / identity / 'workspace'
        after, parent = tree_hashes(child), tree_hashes(manager.workspace)
        baseline = record['baseline_sha256']
        if any(parent.get(p) != baseline.get(p) for p in changes):
            raise ValueError('parent_baseline_conflict; no integration performed')
        if any(after.get(p) != record['after_sha256'].get(p) for p in changes):
            raise ValueError('child_artifact_changed_after_review')
        # Resolve every destination before the first write, including parent links.
        for path in changes:
            target = manager.workspace / path
            if target.is_symlink() or not target.resolve().is_relative_to(manager.workspace.resolve()):
                raise ValueError('integration_path_escape')
        intent = {'id': uuid.uuid4().hex, 'status': 'intent', 'paths': changes,
                  'baseline_sha256': {p: baseline.get(p) for p in changes}}
        record['integration'] = intent
        manager.save()
        journal = manager.directory / 'children' / identity / 'integration.json'
        atomic_json(journal, intent)
        for path in changes:
            target, source = manager.workspace / path, child / path
            if path in after:
                atomic_bytes(target, source.read_bytes())
            else:
                target.unlink()
        record['integration'] = {**intent, 'status': 'integrated', 'main_verification_required': True}
        atomic_json(journal, record['integration'])
        manager.save()
        return record['integration']
