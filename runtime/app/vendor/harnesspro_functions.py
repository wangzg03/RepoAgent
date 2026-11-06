"""Verbatim functions from HarnessPro 8ae622eb; see REFERENCE_SOURCES.json and LICENSE_HARNESSPRO."""
from __future__ import annotations
from typing import Any

def _stage_context(count: int, selection: dict[str, Any]) -> str:
    economy = int(selection.get("economy_after_tools") or 0)
    strict = int(selection.get("strict_after_tools") or 0)
    reminder = int(selection.get("strict_reminder_every") or 8)
    if count == economy:
        return (
            "Economy phase starts now. Batch independent ls/rg/read/test checks, suppress "
            "process narration, and do not repeat unchanged checks. Do not trade away the "
            "behavioral validation or exact final-state checks required by the task."
        )
    if count >= strict and (count == strict or (count - strict) % reminder == 0):
        return (
            "Convergence phase: stop broad exploration. Only inspect evidence directly tied to the "
            "current failure or deliverable. If a concrete failure, unexplained instability, or "
            "mutated deliverable remains, repair it rather than finalizing because the phase changed. "
            "Avoid irreversible cleanup or broad state changes not required by the task, and keep "
            "temporary verification artifacts separate from final deliverables. "
            "Then recheck exact paths, interfaces, behavior, and final state once and finish."
        )
    return ""


def _stage_name(count: int, selection: dict[str, Any]) -> str:
    finalize = int(selection.get("finalize_after_tools") or 0)
    strict = int(selection.get("strict_after_tools") or 0)
    economy = int(selection.get("economy_after_tools") or 0)
    if finalize and count > finalize:
        return "finalize"
    if strict and count >= strict:
        return "strict"
    if economy and count >= economy:
        return "economy"
    return "normal"


def _compact_text(text: str, *, head_chars: int, tail_chars: int) -> str:
    lines = text.splitlines()
    deduped: list[str] = []
    index = 0
    while index < len(lines):
        next_index = index + 1
        while next_index < len(lines) and lines[next_index] == lines[index]:
            next_index += 1
        repeats = next_index - index
        deduped.append(lines[index])
        if repeats > 2:
            deduped.append(f"[previous line repeated {repeats - 1} more times]")
        elif repeats == 2:
            deduped.append(lines[index])
        index = next_index
    compact = "\n".join(deduped)
    if len(compact) <= head_chars + tail_chars:
        return compact
    omitted = len(compact) - head_chars - tail_chars
    return (
        compact[:head_chars]
        + f"\n... [{omitted} characters omitted; use the raw path for a targeted range] ...\n"
        + compact[-tail_chars:]
    )
