"""Deterministic request policy inspired by HarnessPro appendix D; no model calls."""

import copy
import hashlib
import json

from .reference_runtime import budget_stage, compact_observation

POLICY_VERSION = "checkpoint-retention-7"


def compact_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def diagnostic_anchors(groups):
    """Keep early diagnosis and latest probes across every projection, not only recovery.

    Execution labels are observations, not a claim of read-only safety or truth.
    Full intermediate observations remain pageable from the canonical trajectory.
    """
    candidates = [i for i, group in enumerate(groups) if any(
        m.get("extra", {}).get("observation_kind") == "execution"
        and not m.get("extra", {}).get("workspace_changed")
        and not m.get("extra", {}).get("verification_candidate") for m in group[1:])]
    return set(candidates[:2] + candidates[-2:])


def reduce_outputs(groups, retained, max_chars=6000):
    """Reduce older successful bash results; keep diagnostic anchors and recent pairs intact.

    References always point to a retained later result or a paged canonical turn.
    Native calls/reasoning, failed/unknown-status and structured outputs stay intact.
    """
    projected = {i: copy.deepcopy(groups[i]) for i in retained}
    changes, representatives = [], {}
    protected = diagnostic_anchors(groups)
    for flag in ("workspace_changed", "verification_candidate"):
        matches = [i for i, g in enumerate(groups) if any(m.get("extra", {}).get(flag) for m in g[1:])]
        if matches:
            protected.add(matches[-1])
    if not max_chars:
        return projected, changes
    for index in sorted(retained, reverse=True):
        group = projected[index]
        calls = {c["id"]: c["function"] for c in group[0].get("tool_calls", [])}
        for message in group[1:]:
            function = calls.get(message.get("tool_call_id"), {})
            text = message.get("content") or ""
            if function.get("name") != "bash" or message.get("extra", {}).get("returncode") != 0:
                continue
            # Include exact command and output; identical text from different actions is not reuse evidence.
            action_id = message.get("extra", {}).get("action_id")
            header = f"[Captured output: trajectory_read action=output action_id={action_id} offset=0 limit=4000]"
            comparable = text.replace(header, "[Captured output receipt]", 1) if action_id else text
            identity = hashlib.sha256((compact_json(function) + "\0" + comparable).encode()).hexdigest()
            later = representatives.setdefault(identity, index + 1)
            if index >= len(groups) - 2 or index in protected:
                continue
            recall = (f"trajectory_read action=output action_id={action_id} offset=0 limit=4000" if action_id else
                      f"trajectory_read action=context turn={index + 1} offset=0 limit=4000")
            reduced, method = text, "repeated_lines"
            if later != index + 1 and len(text) > 768:
                reduced = f"[Exact command/output duplicate of retained turn {later}; historical observation only. Original: {recall}]"
                method = "exact_duplicate"
            elif len(reduced) > max_chars:
                marker = f"\n[Middle omitted; original {len(text)} chars, sha256={hashlib.sha256(text.encode()).hexdigest()[:16]}; {recall}]\n"
                reduced = compact_observation(text, max_chars - len(marker) - 120) + marker
                method = "bounded_history_output"
            if len(reduced) < len(text):
                message["content"] = reduced
                changes.append({"turn": index + 1, "tool_call_id": message.get("tool_call_id"),
                                "method": method, "original_chars": len(text), "view_chars": len(reduced),
                                "sha256": hashlib.sha256(text.encode()).hexdigest(), "recall": recall})
    return projected, changes


def token_components(messages, schema_tokens=0):
    """Approximate prompt attribution, not a provider tokenizer or billing receipt."""
    from .context import estimate_tokens

    result = {"instructions_and_task": 0, "assistant_text": 0, "native_reasoning": 0,
              "tool_arguments": 0, "tool_observations": 0, "schemas_and_schema_margin": schema_tokens,
              "message_margin": len(messages) * 8}
    for m in messages:
        if m["role"] == "tool":
            label = "tool_observations"
        elif m["role"] == "assistant":
            label = "assistant_text"
        else:
            label = "instructions_and_task"
        result[label] += estimate_tokens([{"content": m.get("content")}])
        result["native_reasoning"] += estimate_tokens([{"reasoning_content": m.get("reasoning_content")}])
        result["tool_arguments"] += estimate_tokens([{"tool_calls": m.get("tool_calls")}])
    return result


class RuntimeGuidance:
    """Numeric budget feedback. Advisory only: never hides tools or changes caps/effort."""
    def __init__(self, environment, llm, settings):
        self.environment, self.llm, self.settings = environment, llm, settings

    def __call__(self):
        budget = getattr(self.llm, "budget", None)
        settings = self.settings
        usage = {"tools_remaining": max(0, settings["tool_limit"] - self.environment.actions)}
        pressures = [self.environment.actions / settings["tool_limit"]]
        if budget is not None:
            shared = getattr(budget, "shared", budget)
            with shared._lock:
                for label, limit_key, spent_key, reserved_key in [
                    ("requests_remaining", "request_limit", "requests", None),
                    ("tokens_remaining", "token_limit", "tokens_spent", "tokens_reserved"),
                    ("usd_remaining", "limit", "spent", "reserved")]:
                    limit = getattr(shared, limit_key, None)
                    if limit is not None:
                        used = getattr(shared, spent_key, 0) + (getattr(shared, reserved_key, 0) if reserved_key else 0)
                        usage[label] = round(max(0, limit - used), 6)
                        pressures.append(used / limit if limit else 1)
            if getattr(budget, "child_limit", None) is not None:
                usage["child_requests_remaining"] = max(0, budget.child_limit - budget.local_requests)
                pressures.append(budget.local_requests / budget.child_limit)
            elif shared is not budget and 'request_limit' in vars(budget) and 'requests' in vars(budget):
                # NativeProvider uses ChildBudget; the injected provider seam uses
                # BudgetGate. Both must see the child's tighter request cap.
                cap, spent = budget.request_limit, budget.requests
                usage['child_requests_remaining'] = max(0, cap - spent)
                pressures.append(spent / cap if cap else 1)
        from .runtime import remaining_time
        seconds = remaining_time()
        if seconds is not None:
            usage["seconds_remaining"] = round(max(0, seconds))
            pressures.append(1 - seconds / settings["agent_seconds"])
        pressure = max(pressures)
        if pressure < .33:
            return None
        stage = budget_stage(pressure)
        if pressure >= .8:
            advice = "Focus on the current repair and final verification. Preserve unresolved failures; report incomplete if needed."
        else:
            advice = "Prefer targeted reads and independent batched checks; reuse unchanged results. Continue necessary repair and verification."
        return {"budget": usage, "guidance": advice, "stage": stage['stage'],
                "pressure": round(pressure, 4), "advisory_only": True}
