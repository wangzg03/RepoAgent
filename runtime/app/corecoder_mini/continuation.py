"""Receipt-grounded next-turn steering, adapted from pi's continuation hook.

This observes work, never chooses a patch, claims task correctness, changes the
tool loadout, replays actions or creates another model request.
"""

import hashlib
import json

from .runtime import check_cancelled
from .task_state import atomic_json


class ActionContinuation:
    def __init__(self, progress, registry, budget_provider=None):
        self.progress, self.registry = progress, registry
        self.budget_provider = budget_provider

    def __call__(self):
        check_cancelled()
        progress, tools = self.progress, self.registry.tools
        data = progress.data
        if not data['observations']:
            return None
        current = data['current']
        version = hashlib.sha256(json.dumps(current, sort_keys=True).encode()).hexdigest()
        recent = data['recent_actions']
        intent = progress.state.requested_kind() if progress.state else None
        read_only = intent in {'explain', 'locate', 'review'}
        failures = progress._blocking_failures()
        check_failures = [r for r in failures if r.get('kind', 'check') == 'check']
        failure_versions = {r['action_id']: r.get('workspace_sha256') for r in recent}
        retry_after_change = bool(check_failures) and all(
            (failed_version := r.get('workspace_sha256') or failure_versions.get(r['action_id']))
            and failed_version != version for r in check_failures)
        edit_errors = [r for r in data['unresolved_commands'] if r.get('kind') == 'edit']
        checked = [r['action_id'] for r in recent if r.get('check_candidate') and r['returncode'] == 0
                   and r.get('workspace_sha256') == version and not data['observation_error']]
        targets = {}
        for receipt in reversed(recent):
            if receipt['returncode'] or receipt.get('kind') not in {'inspection', 'search'}:
                continue
            for path in receipt.get('paths', []):
                if path in current and len(path) <= 240:
                    targets.setdefault(path, {'path': path, 'action': receipt['action_id'],
                                              'current_sha256': current[path]})
        budget = self.budget_provider() if self.budget_provider else None
        phase = data['recovery'].get('active_window') or {}
        converging = phase.get('phase') in {'reanchor', 'converge'} or (budget or {}).get('stage') in {'strict', 'finalize'}
        if data['observation_error']:
            status = 'inspect_state'
            instruction = 'Workspace monitoring failed. Inspect the current target/check; do not claim verified progress from stale receipts.'
        elif read_only:
            status = 'deliver_findings'
            instruction = 'Deliver the requested supported findings; do not edit source for this read-only task.'
        elif edit_errors:
            status = 'repair_edit'
            instruction = ('The edit failed. Correct the stale/missing/ambiguous anchor using the captured target window; '
                           'read that window only if missing or stale, then retry. Do not repeat the same failed arguments.')
        elif retry_after_change:
            status = 'verify_current'
            instruction = ('Source changed since the recorded failure. Rerun that failed execution/check against this '
                           'version now. Its failure stays unresolved until the check succeeds; do not keep editing without retesting.')
        elif failures:
            if any(r.get('kind', 'check') == 'check' for r in failures):
                status = 'repair_failure'
                instruction = ('Repair the recorded failed test/assertion using its actual output; rerun that check. '
                               'A different successful probe does not resolve that failure. Avoid a new repository survey.')
            else:
                status = 'repair_execution'
                instruction = ('A tool invocation repeatedly failed. Fix its syntax/environment or replace the invocation '
                               'with an appropriate available tool. Do not repeat unchanged failing commands. '
                               'This is not evidence that the requested code behavior failed; continue the scoped change.')
        elif data['changed_paths'] and checked:
            status = 'review_delivery'
            instruction = ('A check candidate passed on this version. Review the diff and requested edge cases, then submit '
                           'when the work is covered. This receipt alone does not prove task correctness.')
        elif data['changed_paths']:
            status = 'verify_current'
            instruction = ('Source is changed. Run a relevant real test/assertion against this version; repair failures '
                           'before submitting. Earlier-version checks are stale.')
        elif search := progress.search_feedback():
            status = 'change_search_strategy'
            instruction = search['instruction']
        elif targets or data['observations'] >= 4:
            status = 'implement_or_discriminate'
            instruction = ('Implementation is still missing. For requested code changes, act on the located evidence: '
                           'make a scoped edit, or resolve one concrete uncertainty with a targeted experiment and then edit. '
                           'State that uncertainty with the action; do not restart broad exploration.')
            if intent is None:
                instruction += ' For read-only work, deliver findings instead.'
            elif intent in {'fix', 'implement', 'refactor'} and converging:
                instruction = ('No implementation has been made. Stop expanding the repository survey. '
                               'Use the captured target to make the smallest supported change now, then run its relevant check. '
                               'If a concrete missing fact prevents a safe edit, identify that fact and resolve it with one focused action; '
                               'explain how its result determines the change. If blocked, report the limitation and submit partial work.')
        else:
            status = 'locate_target'
            instruction = 'Locate the relevant implementation/test with a focused search, then act on that target.'
        if intent is None and status in {'repair_edit', 'repair_failure', 'repair_execution', 'change_search_strategy',
                                         'verify_current', 'review_delivery', 'locate_target'}:
            instruction += ' Follow the original request; do not introduce code changes for read-only work.'
        record = {'schema': 1, 'status': status, 'instruction': instruction, 'workspace_sha256': version,
                  'facts': {'intent': intent or 'unspecified',
                            'intent_source': progress.state.data.get('task_contract', {}).get('source', 'model_or_unknown') if progress.state else 'unknown',
                            'source_changes': len(data['changed_paths']),
                            'successful_check_candidates_on_current_version': checked[-2:],
                            'failed_actions': [r['action_id'] for r in (edit_errors or failures)[-2:]],
                            'located_sources': list(targets.values())[:2],
                            'observations_without_source_change': data['actions_without_workspace_change']},
                  'tools': {'edit': 'edit_file' if 'edit_file' in tools else 'bash',
                            'check': 'bash' if 'bash' in tools else None,
                            'recall': 'trajectory_read' if 'trajectory_read' in tools else 'captured history'},
                  'work_checklist': {'source_location_observed': bool(targets),
                                     'source_change_observed': bool(data['changed_paths']),
                                     'current_check_candidate_observed': bool(checked)},
                  'acceptance': False, 'advisory_only': True}
        if status == 'change_search_strategy':
            record['search_observation'] = {k: v for k, v in search.items() if k != 'instruction'}
            # An existing operand is a concrete localization hint, not a claim
            # that its contents were read, a forced tool gate or a fabricated edit.
            paths = [p for p in search['target_paths'] if p in current]
            if paths and 'read_file' in tools:
                record['next_action_hint'] = {'tool': 'read_file', 'arguments': {'file_path': paths[0]}}
        if budget:
            record['budget'] = {'stage': budget['stage'], **budget['budget']}
            if budget['stage'] == 'finalize':
                record['instruction'] += (' Budget is nearly exhausted: focus on this action, a current check and submission. '
                                          'Do not claim success with unresolved failures; report incomplete if necessary.')
        # No source text, model reasoning, private drafts or inferred root cause.
        atomic_json(progress.directory / 'private/action-continuation.json', record)
        return record
