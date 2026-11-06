"""Offline public-fixture comparison against a frozen local source snapshot. No inference.

Run from app: python -m scripts.context_ablation --baseline-archive ../.tmp/.../source.zip --output ../.tmp/...
This checks preservation contracts and estimated prompt size, NOT model pass rate or billed savings.
"""

import argparse
import copy
import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path

from corecoder_mini.context import estimate_tokens
from corecoder_mini.context_view import ContextView, TrajectoryTool, message_groups
from corecoder_mini.delivery import digest
from corecoder_mini.events import Events
from corecoder_mini.state_tools import ScratchpadTool, WorkingState
from corecoder_mini.task_state import atomic_json


def public_cases():
    cases = []
    for kind in ["short", "repeated_log", "repeated_source", "unique_source", "failure_middle", "user_steering", "unicode"]:
        messages = [{"role": "system", "content": "Public development fixture. Preserve the task and current failures."},
                    {"role": "user", "content": "Implement the requested behavior and verify it."}]
        for turn in range(1, 9):
            if kind == "short":
                output = f"step {turn}: small observation"
            elif kind == "repeated_log":
                output = f"start {turn}\n" + "test group passed without warnings\n" * 230 + "end\n"
            elif kind == "unicode":
                output = "\n".join(f"第{i}行：源码与路径 src/sample.py，轮次{turn}。" for i in range(340))
            else:
                output = "\n".join(f"value_{i} = {i}  # public example {0 if kind == 'repeated_source' else turn}" for i in range(240))
            if kind == "failure_middle" and turn == 3:
                output += "\nCOUNTEREXAMPLE_IN_MIDDLE\n" + "preserve failure details\n" * 200
            if kind == "user_steering" and turn == 6:
                messages.append({"role": "user", "content": "New requirement: preserve all Unicode inputs."})
            command = "cat src/sample.py" if kind == "repeated_source" else f"inspect src/sample.py --range={turn}"
            messages.extend([{"role": "assistant", "content": "Inspect current source", "reasoning_content": "native-protocol-state",
                              "tool_calls": [{"id": str(turn), "type": "function", "function": {
                                  "name": "bash", "arguments": json.dumps({"command": command})}}]},
                             {"role": "tool", "tool_call_id": str(turn), "content": output,
                              "extra": {"returncode": 1 if kind == "failure_middle" and turn == 3 else 0}}])
        cases.append({"id": kind, "messages": messages, "note": "Current hypothesis and next experiment must remain available."})
    return cases


def compare(baseline_archive, output):
    output, baseline_archive = Path(output), Path(baseline_archive)
    output.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(baseline_archive) as archive:
        source = archive.read("app/corecoder_mini/context_view.py")
    # This is executable project code: only use a trusted local source snapshot, never a downloaded archive.
    module_path = output / "_baseline_context.py"
    module_path.write_bytes(source)
    spec = importlib.util.spec_from_file_location("corecoder_mini._frozen_context_baseline", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fixtures = public_cases()
    atomic_json(output / "frozen-cases.json", fixtures)  # Freeze before evaluating either policy.
    settings = {"target_tokens": 10000, "keep_recent": 4, "capacity": 15000}
    atomic_json(output / "frozen-comparison.json", {
        "fixture_sha256": digest(output / "frozen-cases.json"), "baseline_archive_sha256": digest(baseline_archive),
        "baseline_module_sha256": hashlib.sha256(source).hexdigest(),
        "candidate_modules": {p.name: digest(p) for p in [Path(__file__).parents[1] / "corecoder_mini/context_view.py",
                                                         Path(__file__).parents[1] / "corecoder_mini/token_policy.py"]},
        "settings": settings, "provider_calls": 0, "fixture_scope": "synthetic_public_development_only"})
    rows = []
    for case in fixtures:
        measurements = {}
        for name, cls in [("baseline", module.ContextView), ("candidate", ContextView)]:
            directory = output / case["id"] / name
            state = WorkingState(directory)
            ScratchpadTool(state).execute("write", "hypothesis", case["note"])
            projector = cls(directory, state, Events(directory), target_tokens=settings["target_tokens"],
                            keep_recent=settings["keep_recent"])
            original = copy.deepcopy(case["messages"])
            try:
                view, receipt = projector.project(original, settings["capacity"])
            except RuntimeError as exc:
                if "context_capacity" not in str(exc):
                    raise
                measurements[name] = {"error": "context_capacity", "estimated_message_tokens": None,
                                      "preservation_and_recall_ok": False}
                continue
            assert original == case["messages"]
            assert json.loads((directory / "canonical-messages.json").read_text(encoding="utf-8")) == original
            _, projected = message_groups(view)
            _, groups = message_groups(original)
            by_id = {g[0]["tool_calls"][0]["id"]: g for g in projected}
            for turn in [str(len(groups) - 1), str(len(groups))]:
                assert by_id[turn][1]["content"] == groups[int(turn) - 1][1]["content"]
            for index, group in enumerate(groups):
                if group[1]["extra"]["returncode"] != 0:
                    assert by_id[str(index + 1)][1]["content"] == group[1]["content"]
            encoded = json.dumps(view, ensure_ascii=False)
            assert case["note"] in encoded and original[1]["content"] in encoded
            for index, group in enumerate(groups):
                for offset in range(0, len(group[1]["content"]), 4000):
                    recovered = json.loads(TrajectoryTool(directory).execute(turn=index + 1, offset=offset, limit=4000))
                    assert recovered[0]["observations"][0]["content"] == group[1]["content"][offset:offset + 4000]
            measurements[name] = {"estimated_message_tokens": estimate_tokens(view),
                                  "retained_turns": receipt["retained_turns"], "preservation_and_recall_ok": True,
                                  "reduced_observations": len(receipt.get("output_reductions", []))}
        old = measurements["baseline"]["estimated_message_tokens"]
        new = measurements["candidate"]["estimated_message_tokens"]
        rows.append({"case": case["id"], **measurements,
                     "estimated_reduction_percent": round((old - new) / old * 100, 2) if old and new else None})
    paired = [r for r in rows if r["estimated_reduction_percent"] is not None]
    before = sum(r["baseline"]["estimated_message_tokens"] for r in paired)
    after = sum(r["candidate"]["estimated_message_tokens"] for r in paired)
    report = {"kind": "offline_context_contract_and_size_comparison", "cases": rows,
              "aggregate_baseline_estimated_message_tokens": before, "aggregate_candidate_estimated_message_tokens": after,
              "aggregate_estimated_reduction_percent": round((before - after) / before * 100, 2) if before else None,
              "paired_cases": len(paired),
              "capacity_failures": {name: [r["case"] for r in rows if r[name].get("error")] for name in ("baseline", "candidate")},
              "provider_calls": 0, "quality_scope": "preservation/recall contracts only; model correctness not evaluated",
              "billing_scope": "estimated message sizes only; unchanged schemas, output tokens and runtime reminders excluded",
              "automatic_policy_promotion": False}
    atomic_json(output / "result.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.baseline_archive, args.output)
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}))


if __name__ == "__main__":
    main()
