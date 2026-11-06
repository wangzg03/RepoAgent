"""Official grading integration outside the agent; hidden tests never reach inference."""

import shutil
import subprocess
import time
import uuid
from pathlib import Path

from scripts.swebench_io import repository_metadata

from .delivery import atomic_bytes, bind_verdict, digest, read_json
from .task_state import atomic_json

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def linux(path):
    path = Path(path).resolve()
    if not path.drive or len(path.drive) != 2:
        raise ValueError("WSL adapter requires a Windows drive path")
    return "/mnt/" + path.drive[0].lower() + path.as_posix()[2:]


def prepare(directory, config, task_id):
    """Reuse pinned public-image/scorer assets, never historical generated patches."""
    directory, assets = Path(directory), Path(config["assets_directory"]).resolve(strict=True)
    rows = read_json(assets / "private/judge.json")
    selected = [r for r in rows if r["instance_id"] == task_id]
    if len(selected) != 1:
        raise ValueError("Missing or duplicate official instance")
    official = directory / "official"
    atomic_json(official / "private/judge.json", selected)
    environment = assets / "official-environment.json"
    if not environment.is_file():
        raise ValueError("Frozen official scorer environment required")
    shutil.copyfile(environment, official / "official-environment.json")
    archive = assets / "private/git" / (task_id + ".tar.gz")
    if not archive.is_file():
        raise ValueError("Original public repository Git context is missing")
    metadata = repository_metadata(Path(config["baseline_source"]))
    if not metadata.is_file():
        raise ValueError("Original repository metadata missing")
    shutil.copyfile(metadata, official / "repository-metadata.json")
    original = read_json(assets / "config.json")
    atomic_json(official / "config.json", {"wheelhouse_linux": original["wheelhouse_linux"],
                "baseline_metadata_sha256": {task_id: digest(official / "repository-metadata.json")}})
    return {"kind": "swebench_official", "linux_python": config["linux_python"],
            "score_seconds": config.get("score_seconds", 1800), "task_id": task_id,
            "archive": str(archive), "archive_sha256": digest(archive),
            "official_files_sha256": {p.relative_to(official).as_posix(): digest(p)
                for p in official.rglob("*") if p.is_file()}}


def verify(directory, config):
    directory = Path(directory)
    official = directory / "official"
    for name, expected in config["official_files_sha256"].items():
        if digest(official / name) != expected:
            raise ValueError("Frozen official inputs drift")
    if digest(config["archive"]) != config["archive_sha256"]:
        raise ValueError("Original public repository Git context drift")
    frozen = read_json(directory / "frozen.json")
    result = read_json(directory / "result.json")
    from .supervisor import memory_status
    # Give the owned model unloading a bounded grace period; keep the resource guard.
    stop = time.monotonic() + 30
    while True:
        available, load = memory_status()
        if available >= 5 and load < 85:
            break
        if time.monotonic() >= stop:
            raise RuntimeError("official_scoring_resource_unavailable")
        time.sleep(.5)
    job = uuid.uuid4().hex
    score_directory = official / "private/scores" / job
    row = read_json(official / "private/judge.json")[0]
    request = {"output": linux(official), "task_id": config["task_id"], "job_id": job,
               "score_directory": linux(score_directory), "base_directory": linux(directory / "baseline"),
               "image_id": frozen["image_id"], "score_seconds": config["score_seconds"],
               "patch_path": linux(directory / "model.patch"), "base_commit": row["base_commit"],
               "repository_git_archive": linux(config["archive"]), "repository_git_sha256": config["archive_sha256"],
               "repository_metadata": linux(official / "repository-metadata.json")}
    request_path = official / ("request-" + job + ".json")
    atomic_json(request_path, request)
    # No shell interpolation, no model access to the judge or this subprocess.
    command = ["wsl.exe", "-d", "Ubuntu", "--", config["linux_python"], linux(SCRIPTS / "swebench_official.py"),
               "--request", linux(request_path)]
    try:
        with (official / ("score-" + job + ".log")).open("wb") as log:
            subprocess.run(command, stdout=log, stderr=log, timeout=config["score_seconds"] + 120, check=True)
    finally:
        from .supervisor import cleanup_owned
        cleanup_owned(job)
    scored_patch = score_directory / "prediction.patch"
    if digest(scored_patch) != result["patch_sha256"]:
        raise ValueError("Official scorer evaluated different patch bytes")
    raw = read_json(score_directory / "result.json")
    report = read_json(score_directory / "official-report.json")
    if raw["resolved"] != report[config["task_id"]]["resolved"]:
        raise ValueError("Official result/report disagree")
    verdict = {**raw, "patch_sha256": digest(scored_patch),
               "official_report_sha256": digest(score_directory / "official-report.json")}
    evidence = directory / "verification.json"
    atomic_json(evidence, verdict)
    atomic_bytes(directory / "official-report.json", (score_directory / "official-report.json").read_bytes())
    return bind_verdict(directory, verdict, evidence, kind="swebench_official")
