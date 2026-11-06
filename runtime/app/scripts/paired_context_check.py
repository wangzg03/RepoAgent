"""Isolated source-snapshot projection check; no inference or correctness/cost claim."""

import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path

from corecoder_mini.delivery import digest, read_json
from corecoder_mini.task_state import atomic_json
from scripts.context_ablation import public_cases

RUNNER = r'''
import json,sys,copy
from pathlib import Path
from corecoder_mini.context_view import ContextView,TrajectoryTool,message_groups
from corecoder_mini.state_tools import WorkingState,ScratchpadTool
from corecoder_mini.supervisor import validate_extensions
from corecoder_mini.events import Events
from corecoder_mini.context import estimate_tokens
from corecoder_mini.task_state import atomic_json
fixtures=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
root=Path(sys.argv[2]); settings={};validate_extensions(settings)
rows=[]
for case in fixtures:
 directory=root/case['id']; state=WorkingState(directory)
 ScratchpadTool(state).execute('write','hypothesis',case['note'])
 view=ContextView(directory,state,Events(directory),target_tokens=settings['history_target_tokens'],
                  keep_recent=settings['keep_recent_turns'],history_output_chars=settings['history_output_chars'])
 original=copy.deepcopy(case['messages'])
 try:wire,receipt=view.project(original,15000)
 except RuntimeError as exc:
  if 'context_capacity' not in str(exc):raise
  rows.append({'id':case['id'],'error':'context_capacity'});continue
 assert original==case['messages']
 _,groups=message_groups(wire);_,full=message_groups(original)
 kept={g[0]['tool_calls'][0]['id']:g for g in groups}
 for i,g in enumerate(full):
  if i>=len(full)-2 or g[1]['extra']['returncode']!=0:
   assert kept[str(i+1)][1]['content']==g[1]['content']
  for offset in range(0,len(g[1]['content']),4000):
   recovered=json.loads(TrajectoryTool(directory).execute(turn=i+1,offset=offset,limit=4000))
   assert recovered[0]['observations'][0]['content']==g[1]['content'][offset:offset+4000]
 assert case['note'] in json.dumps(wire) and original[1]['content'] in json.dumps(wire)
 rows.append({'id':case['id'],'estimated_message_tokens':estimate_tokens(wire),'preservation_and_recall':True})
atomic_json(root/'result.json',{'module_path':str(Path(sys.modules[ContextView.__module__].__file__).resolve()),
                              'policy_settings':settings,'rows':rows,'provider_calls':0})
'''


def run(baseline_archive, output):
    baseline_archive, output = Path(baseline_archive).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    baseline_app = output / "baseline-source/app"
    with zipfile.ZipFile(baseline_archive) as archive:
        for member in archive.infolist():
            parts = Path(member.filename).parts
            if not parts or parts[0] != "app" or member.is_dir():
                continue
            if ".." in parts or Path(member.filename).is_absolute() or ":" in member.filename or "\\" in member.filename:
                raise ValueError("unsafe_snapshot_path")
            if member.file_size > 10_000_000:
                raise ValueError("snapshot_member_too_large")
            destination = output / "baseline-source" / member.filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(member))
    fixtures = output / "frozen-cases.json"
    atomic_json(fixtures, public_cases())
    candidate_app = Path(__file__).resolve().parents[1]
    atomic_json(output / "frozen.json", {"baseline_archive_sha256": digest(baseline_archive),
                "fixtures_sha256": digest(fixtures), "scope": "public_synthetic_context_contracts",
                "candidate_source_sha256": {p.relative_to(candidate_app).as_posix(): digest(p)
                    for p in (candidate_app / "corecoder_mini").rglob("*.py")}, "provider_calls": 0})
    results = {}
    for label, app in (("baseline", baseline_app), ("candidate", candidate_app)):
        subprocess.run([sys.executable, "-B", "-c", RUNNER, str(fixtures), str(output / label)],
                       cwd=app, capture_output=True, check=True, timeout=60)
        results[label] = read_json(output / label / "result.json")
        if not Path(results[label]["module_path"]).is_relative_to(app):
            raise ValueError("source_isolation_failed")
    pairs, failures = [], []
    for old, new in zip(results["baseline"]["rows"], results["candidate"]["rows"]):
        assert old["id"] == new["id"]
        if "error" in old or "error" in new:
            failures.append({"id": old["id"], "baseline": old.get("error"), "candidate": new.get("error")})
        else:
            pairs.append({"id": old["id"], "baseline_tokens_estimated": old["estimated_message_tokens"],
                          "candidate_tokens_estimated": new["estimated_message_tokens"]})
    before = sum(p["baseline_tokens_estimated"] for p in pairs)
    after = sum(p["candidate_tokens_estimated"] for p in pairs)
    report = {"kind": "offline_source_isolated_projection_check", "pairs": pairs, "capacity_failures": failures,
              "source_isolation_verified": True, "provider_calls": 0,
              "aggregate_baseline_estimated_tokens": before, "aggregate_candidate_estimated_tokens": after,
              "estimated_reduction": 1 - after / before if before else None,
              "real_model_quality_measured": False, "real_cost_measured": False,
              "forty_percent_cost_quality_target_proven": False}
    atomic_json(output / "result.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-archive", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.baseline_archive, args.output)
    print(json.dumps({k: v for k, v in report.items() if k not in {"pairs", "capacity_failures"}}))
