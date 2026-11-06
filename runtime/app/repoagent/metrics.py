"""Observed metrics only; localization labels stay on the evaluator side."""
import math
import statistics
import time
from pathlib import Path

from corecoder_mini.delivery import read_json
from corecoder_mini.task_state import atomic_json


def localization(index, cases, k=5):
    if not 1 <= k <= 30:
        raise ValueError('invalid_k')
    rows = []
    for case in cases:
        expected = set(case['expected_paths'])
        if not expected:
            raise ValueError('nonempty_evaluator_labels_required')
        start = time.monotonic()
        result = index.search(case['query'], case.get('mode', 'auto'), limit=30)
        paths = list(dict.fromkeys(hit['path'] for hit in result['hits']))[:k]
        relevant = [path in expected for path in paths]
        found = set(paths) & expected
        dcg = sum(int(valid) / math.log2(rank + 2) for rank, valid in enumerate(relevant))
        ideal = sum(1 / math.log2(rank + 2) for rank in range(min(k, len(expected))))
        rows.append({'id': case['id'], 'paths': paths, 'recall_at_k': len(found) / len(expected),
                     'precision_at_k': len(found) / k,
                     'mrr': next((1/(n+1) for n, valid in enumerate(relevant) if valid), 0),
                     'ndcg': dcg / ideal, 'milliseconds': (time.monotonic()-start)*1000,
                     'incomplete_index': result['index']['incomplete']})
    return {'schema': 1, 'k': k, 'cases': rows, 'case_count': len(rows),
            'aggregation': {name: statistics.mean(row[name] for row in rows) if rows else None
                            for name in ['recall_at_k', 'precision_at_k', 'mrr', 'ndcg', 'milliseconds']},
            'interpretation': 'file localization on supplied labeled cases; not repair correctness'}


def summarize(run):
    run = Path(run)
    result, frozen = read_json(run / 'result.json', {}), read_json(run / 'frozen.json', {})
    tools = [read_json(p, {}) for p in (run / 'agent/actions').glob('*.json')]
    events = [read_json(p, {}) for p in (run / 'agent/events').glob('*.json')]
    first = min((e.get('time', float('inf')) for e in events if e.get('type') == 'session_started'), default=None)
    metrics = {'task_id': result.get('task_id'), 'state': result.get('state'),
               'model_submitted': result.get('model_submitted', False),
               'artifact_correct': result.get('artifact_correct'), 'official_correct': result.get('official_correct'),
               'autonomous_complete': result.get('autonomous_complete', False),
               'verification_kind': result.get('verification_kind'),
               'request_usage': result.get('request_usage', {}),
               'cost_accounting': result.get('cost_accounting', {}),
               'tool_count': len(tools), 'tool_failures': sum(t.get('result', {}).get('returncode', 0) != 0 for t in tools),
               'localization_calls': sum(t.get('tool') == 'code_search' for t in tools),
               'latency_seconds': result.get('time', first) - first if first else None,
               'termination': result.get('checkpoint_exit'), 'source_frozen': bool(frozen.get('source_sha256')),
               'task_contract_recorded': (run / 'agent/task-contract.json').is_file(),
               'metric_targets_are_not_observations': True}
    atomic_json(run.parent / 'metrics.json', metrics)
    return metrics


def aggregate(packets):
    """Keep all frozen IDs in denominators; incomplete official results stay null."""
    rows = []
    for packet in packets:
        packet = Path(packet)
        result = read_json(packet / 'run/result.json', {})
        rows.append({'packet': packet.name, 'started': bool(result) and result.get('state') != 'frozen',
                     'official_correct': result.get('official_correct'),
                     'autonomous_complete': result.get('autonomous_complete', False),
                     'timeout': result.get('terminal', {}).get('reason') == 'task_deadline'
                                or result.get('checkpoint_exit') == 'DeadlineExceeded',
                     'tokens': sum(result.get('request_usage', {}).get(k, 0) for k in
                                   ['reported_prompt_tokens', 'reported_completion_tokens']),
                     'estimated_usd': result.get('cost_accounting', {}).get('known_usage_tariff_usd', 0)})
    planned = len(rows)
    passed = sum(r['official_correct'] is True for r in rows)
    complete_scores = bool(rows) and all(type(r['official_correct']) is bool for r in rows)
    cost = sum(r['estimated_usd'] or 0 for r in rows)
    return {'planned': planned, 'started': sum(r['started'] for r in rows),
            'official_scored': sum(type(r['official_correct']) is bool for r in rows),
            'official_passed': passed, 'official_results_complete': complete_scores,
            'official_correct_rate': passed / planned if complete_scores else None,
            'autonomous_completion_rate': sum(r['autonomous_complete'] for r in rows) / planned if planned else None,
            'task_timeout_rate': sum(r['timeout'] for r in rows) / planned if planned else None,
            'reported_tokens': sum(r['tokens'] for r in rows), 'known_estimated_usd': cost,
            'known_estimated_cost_per_official_pass': cost / passed if passed else None,
            'is_invoice': False, 'cases': rows,
            'pass_at_k_note': 'No pass@k reimplementation; per-case official reports retain upstream scores.'}
