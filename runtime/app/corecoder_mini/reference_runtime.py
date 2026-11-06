"""Small protocol adapters around pinned upstream functions, not new agent loops."""

import importlib
import math
from pathlib import Path

from . import upstream as _upstream  # noqa: F401 - selects the vendored path before imports below
from .delivery import read_json
from .task_state import atomic_json

_harnesspro = importlib.import_module("harnesspro_functions")
_recreate = importlib.import_module("recreate_read_trajectory")
POLICY = read_json(Path(__file__).resolve().parents[1] / "vendor/harnesspro_policy.json")


def compact_observation(text, max_chars):
    """Reuse the released hook's dedup/head-tail function; callers add their recall link."""
    output = POLICY["tool_output"]
    return _harnesspro._compact_text(text, head_chars=max(1, max_chars * output["head_chars"] // output["max_chars"]),
                                    tail_chars=max(1, max_chars * output["tail_chars"] // output["max_chars"]))


def stage_guidance(tool_count):
    # One fixed upstream profile, no model classifier, modes, effort changes or added requests.
    selection = POLICY["profiles"][POLICY["default_profile"]]
    return {"source": "HarnessPro pinned hook", "stage": _harnesspro._stage_name(tool_count, selection),
            "guidance": _harnesspro._stage_context(tool_count, selection)}


def budget_stage(pressure):
    """Adapt the pinned stage selector to the tightest *actual shared* budget.

    No new hard caps, tool denial or provider setting changes. Fixed upstream
    tool counts are retained for reference tests, not used as our runtime clock.
    """
    selection = {'economy_after_tools': 33, 'strict_after_tools': 50, 'finalize_after_tools': 79}
    count = math.floor(max(0, min(1, pressure)) * 100 + 1e-8)
    return {'stage': _harnesspro._stage_name(count, selection),
            'source': 'HarnessPro stage selector, adapted to shared budget utilization'}


def episode_index(directory, groups):
    """Translate OpenAI tool groups into the original ReCreate text trajectory format.

    This directory is private and generated. It never contains expected_tests.txt,
    hidden judge data, other episodes, or private reasoning. Only failed executions
    and test candidates are exposed as feedback, so a source literal is not an error.
    """
    rows, file_actions, file_errors = [], [], []
    import json
    for step, group in enumerate(groups, 1):
        commands = []
        calls = {}
        for call in group[0].get("tool_calls", []):
            try:
                args = json.loads(call["function"].get("arguments", "{}"))
            except (ValueError, TypeError):
                args = {}
            calls[call['id']] = (call['function']['name'], args)
            commands.append(str(args.get("command", "")))
        rows.append({"role": "assistant", "content": "\n".join(commands)})
        for message in group[1:]:
            extra = message.get("extra", {})
            name, args = calls.get(message.get('tool_call_id'), ('', {}))
            if name in {'read_file', 'edit_file', 'write_file'}:
                anchor = {'step': step, 'tool': name, 'path': args.get('file_path'),
                          'action_id': extra.get('action_id'), 'returncode': extra.get('returncode')}
                file_actions.append(anchor)
                if extra.get('returncode') not in {0, None}:
                    file_errors.append({**anchor, 'type': 'FileOperationError'})
            if extra.get("returncode") not in {0, None} or extra.get("verification_candidate"):
                rows.append({"role": "user", "content": str(message.get("content") or "")})
    path = Path(directory) / "private/recreate-episode/trajectory.json"
    atomic_json(path, {"messages": rows, "info": {"exit_status": "in_progress"}})
    result = _recreate.TrajectoryIndexer(path).build_index()
    result.pop("expected", None)  # Upstream optional expected-test facility is never exposed.
    result['file_tool_actions'] = file_actions
    result['errors'].extend(file_errors)
    result['errors'].sort(key=lambda row: row.get('step', 0))
    atomic_json(path.with_name("trajectory_index.json"), result)
    return result
