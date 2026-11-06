"""Read-only paired evaluation: quality first, then cache-aware cost. No inference."""

import argparse
import hashlib
import json
from pathlib import Path

from .accounting import usage_charge, validate_rates
from .delivery import digest, read_json
from .task_state import atomic_json

MATCHED_SETTINGS = ("step_limit", "tool_limit", "request_limit", "token_limit", "dollar_limit",
                    "agent_seconds", "request_seconds", "tool_seconds", "context_tokens",
                    "max_output_tokens", "reasoning_effort", "provider_attempts", "provider_retry_limit")


def collect(directory):
    directory = Path(directory)
    frozen, result = read_json(directory / "frozen.json"), read_json(directory / "result.json")
    if not frozen or not result or frozen["task_id"] != result["task_id"]:
        raise ValueError("missing_or_mismatched_run_identity")
    if digest(directory / "model.patch") != result["patch_sha256"]:
        raise ValueError("artifact_drift")
    evidence = Path(result.get("evidence_path", ""))
    if not evidence.is_file() or digest(evidence) != result.get("evidence_sha256"):
        raise ValueError("missing_or_changed_verification_evidence")
    verdict = read_json(evidence)
    if (verdict.get("patch_sha256") != result["patch_sha256"] or result.get("verification_ok") is not True
            or verdict.get("ok") is not True):
        raise ValueError("unverified_artifact")
    if result.get("artifact_correct") != bool(verdict.get("resolved") and verdict.get("patch_applied")):
        raise ValueError("result_and_verification_disagree")
    submitted = read_json(directory / "agent/submission.json", {}).get("model_submitted") is True
    if (result.get("model_submitted") is not submitted or
            result.get("autonomous_complete") is not bool(submitted and result["artifact_correct"])):
        raise ValueError("submission_and_completion_disagree")
    config = read_json(directory / "verification-config.json")
    if not config or digest(directory / "verification-config.json") != frozen.get("verification_config_sha256"):
        raise ValueError("verification_configuration_drift")
    if config.get("kind") != result.get("verification_kind"):
        raise ValueError("verification_kind_mismatch")
    if result["verification_kind"] == "swebench_official":
        official = directory / "official-report.json"
        if (not official.is_file() or digest(official) != verdict.get("official_report_sha256")
                or read_json(official).get(frozen["task_id"], {}).get("resolved") is not result["artifact_correct"]
                or result.get("official_correct") is not result["artifact_correct"]):
            raise ValueError("original_official_report_required")
    rates = validate_rates(frozen["model"].get("pricing_usd_per_million"))
    paths = list((directory / "agent/requests").glob("*.json"))
    paths += list((directory / "agent/children").glob("*/agent/requests/*.json"))
    charges, unknown, cache_unknown, transports = [], 0, 0, 0
    for path in paths:
        row = read_json(path)
        if row.get("transport_started") is False:
            continue
        transports += 1
        charge = usage_charge({"prompt_tokens": row.get("prompt_tokens"), "completion_tokens": row.get("completion_tokens"),
                               "prompt_cache_hit_tokens": row.get("cached_prompt_tokens")}, rates)
        if row.get("usage_complete") is not True or charge is None:
            unknown += 1
        else:
            charges.append(charge)
            cache_unknown += len(rates) == 3 and not charge["cache_usage_complete"]
    matched = {"task": hashlib.sha256(frozen["public_task"].encode()).hexdigest(),
               "baseline": frozen["baseline_sha256"], "image": frozen["image_id"],
               "model": frozen["model"], "runtime_versions": frozen.get("runtime_versions"),
               "budgets_and_model_settings": {k: frozen["settings"].get(k) for k in MATCHED_SETTINGS},
               "verification": config, "public_submission_check": frozen["settings"].get("public_check_command"),
               "scorer_source": {k: v for k, v in frozen.get("source_sha256", {}).items()
                                 if k.startswith("scripts/swebench_")}}
    return {"task_id": frozen["task_id"], "source": str(directory.resolve()), "matched": matched,
            "official_correct": result.get("official_correct"), "artifact_correct": result.get("artifact_correct"),
            "autonomous_complete": result.get("autonomous_complete") is True,
            "verification_kind": result.get("verification_kind"), "transports": transports,
            "known_tariff_usd": sum(c["cost_usd"] for c in charges), "unknown_usage_requests": unknown,
            "unknown_cache_requests": cache_unknown, "cache_aware_pricing": len(rates) == 3,
            "source_version": frozen.get("source_sha256"), "is_invoice": False}


def compare(baseline, candidate, target_reduction=.4):
    if not 0 < target_reduction < 1:
        raise ValueError("invalid_cost_reduction_target")
    def indexed(paths):
        rows = [collect(p) for p in paths]
        if len({r["task_id"] for r in rows}) != len(rows):
            raise ValueError("duplicate_task_ids")
        return {r["task_id"]: r for r in rows}
    old, new = indexed(baseline), indexed(candidate)
    if not old or old.keys() != new.keys():
        raise ValueError("paired_frozen_task_sets_required_no_dropped_failures")
    reasons, pairs = [], []
    for task_id, incumbent in old.items():
        challenger = new[task_id]
        if incumbent["matched"] != challenger["matched"]:
            raise ValueError("unmatched_model_budget_task_or_evaluator: " + task_id)
        for row in (incumbent, challenger):
            if not row["cache_aware_pricing"]:
                reasons.append("explicit_cache_tariff_required: " + task_id)
            if row["unknown_usage_requests"] or row["unknown_cache_requests"] or not row["transports"]:
                reasons.append("incomplete_usage: " + task_id)
            if row["verification_kind"] != "swebench_official" or type(row["official_correct"]) is not bool:
                reasons.append("not_official_benchmark_evidence: " + task_id)
        if incumbent["artifact_correct"] is True and challenger["artifact_correct"] is not True:
            reasons.append("per_task_quality_regression: " + task_id)
        if incumbent["autonomous_complete"] and not challenger["autonomous_complete"]:
            reasons.append("autonomous_delivery_regression: " + task_id)
        pairs.append({"task_id": task_id, "baseline_correct": incumbent["artifact_correct"],
                      "candidate_correct": challenger["artifact_correct"],
                      "baseline_usd": incumbent["known_tariff_usd"], "candidate_usd": challenger["known_tariff_usd"]})
    before, after = sum(r["known_tariff_usd"] for r in old.values()), sum(r["known_tariff_usd"] for r in new.values())
    reduction = 1 - after / before if before > 0 else None
    if not any(r["autonomous_complete"] and r["artifact_correct"] for r in old.values()):
        reasons.append("baseline_has_no_autonomous_success")
    if reduction is None or reduction + 1e-12 < target_reduction:
        reasons.append("cost_target_not_met")
    return {"schema": 1, "task_count": len(old), "target_reduction": target_reduction,
            "meets_observed_target": not reasons, "reasons": sorted(set(reasons)), "pairs": pairs,
            "baseline_tariff_usd": before, "candidate_tariff_usd": after, "observed_cost_reduction": reduction,
            "baseline_official_passed": sum(r["official_correct"] is True for r in old.values()),
            "candidate_official_passed": sum(r["official_correct"] is True for r in new.values()),
            "baseline_cost_per_official_pass": before / sum(r["official_correct"] is True for r in old.values())
                if any(r["official_correct"] is True for r in old.values()) else None,
            "candidate_cost_per_official_pass": after / sum(r["official_correct"] is True for r in new.values())
                if any(r["official_correct"] is True for r in new.values()) else None,
            "baseline_sources": [r["source_version"] for r in old.values()],
            "candidate_sources": [r["source_version"] for r in new.values()],
            "is_invoice": False, "policy_promoted": False,
            "limitation": "Paired observed results only, not statistical non-inferiority or proof for unseen tasks. No automatic promotion."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", nargs="+", required=True)
    parser.add_argument("--candidate", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("comparison_output_must_be_new")
    report = compare(args.baseline, args.candidate)
    atomic_json(args.output, report)
    print(json.dumps({k: report[k] for k in ("task_count", "meets_observed_target", "observed_cost_reduction", "reasons")}))
    return 0 if report["meets_observed_target"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
