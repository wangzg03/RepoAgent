"""Linux adapter: upstream SWE-bench v4 test script, parser and grading, unchanged."""

import argparse
import hashlib
import inspect
import json
import subprocess
import time
from pathlib import Path


def evaluation_spec(row):
    # Existing official instance images need no environment build script or network
    # fetch. Construct the upstream TestSpec using its ORIGINAL evaluation generator.
    from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS
    from swebench.harness.test_spec.create_scripts import make_eval_script_list
    from swebench.harness.test_spec.test_spec import TestSpec

    specs = MAP_REPO_VERSION_TO_SPECS[row["repo"]][row["version"]]
    script = make_eval_script_list(row, specs, "testbed", "/testbed", row["base_commit"], row["test_patch"])
    return TestSpec(instance_id=row["instance_id"], repo=row["repo"], version=row["version"],
                    repo_script_list=[], env_script_list=[], eval_script_list=script, arch="x86_64",
                    FAIL_TO_PASS=json.loads(row["FAIL_TO_PASS"]), PASS_TO_PASS=json.loads(row["PASS_TO_PASS"]),
                    language="py", docker_specs=specs.get("docker_specs", {}), namespace="swebench")


def source_copy_command():
    # Windows bind mounts report permissive modes. Restore exported executable
    # metadata after this normalization, before comparing with the original Git tree.
    return ("find /source -mindepth 1 -maxdepth 1 -exec cp -a --no-preserve=ownership -t /testbed -- {} + && "
            "find /testbed -type f -exec chmod 0644 -- {} +")


def environment(output):
    import docker
    import swebench

    if swebench.__version__ != "4.1.0":
        raise ValueError("Only pinned official swebench 4.1.0 is accepted")
    root = Path(swebench.__file__).parent
    files = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*.py")}
    package = {"version": swebench.__version__, "source_sha256": files,
               "dependencies": subprocess.check_output([__import__("sys").executable, "-m", "pip", "freeze"], text=True)}
    rows = json.loads((output / "private/judge.json").read_text())
    config = json.loads((output / "config.json").read_text())
    wheelhouse = Path(config["wheelhouse_linux"])
    if not wheelhouse.is_dir():
        raise ValueError("Offline wheelhouse is unavailable")
    package["wheel_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in wheelhouse.glob("*.whl")}
    specs = [evaluation_spec(row) for row in rows]
    package["image_names"] = [s.instance_image_key for s in specs]
    package["docker_ping"] = docker.from_env().ping()
    (output / "official-environment.json").write_text(json.dumps(package, indent=2) + "\n")
    print(json.dumps({"version": package["version"], "validated_tasks": len(rows), "docker_ping": package["docker_ping"]}))


def score(request):
    import docker
    from swebench.harness.grading import get_eval_report

    output, directory = Path(request["output"]), Path(request["score_directory"])
    environment_record = json.loads((output / "official-environment.json").read_text())
    import swebench
    root = Path(swebench.__file__).parent
    for name, expected in environment_record["source_sha256"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Official scorer source drift")
    config = json.loads((output / "config.json").read_text())
    wheelhouse = Path(config["wheelhouse_linux"])
    for name, expected in environment_record["wheel_sha256"].items():
        if hashlib.sha256((wheelhouse / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Offline dependency drift")
    rows = json.loads((output / "private/judge.json").read_text())
    row = next(r for r in rows if r["instance_id"] == request["task_id"])
    spec = evaluation_spec(row)
    directory.mkdir(parents=True, exist_ok=False)
    from swebench_git import restore_git_command, validate_git_archive
    git_archive = Path(request["repository_git_archive"])
    if validate_git_archive(git_archive) != request["repository_git_sha256"]:
        raise ValueError("Frozen private Git context drift")
    import shutil
    shutil.copyfile(git_archive, directory / "repository-git.tar.gz")
    if request.get("control") == "positive":
        patch = row["patch"]
    elif request.get("control") == "negative":
        # A harmless new documentation file exercises application + real failing tests.
        patch = "diff --git a/CORECODER_CONTROL.txt b/CORECODER_CONTROL.txt\nnew file mode 100644\n--- /dev/null\n+++ b/CORECODER_CONTROL.txt\n@@ -0,0 +1 @@\n+negative infrastructure control\n"
    else:
        patch = Path(request["patch_path"]).read_text()
    (directory / "prediction.patch").write_text(patch)
    (directory / "eval.sh").write_text(spec.eval_script)
    client = docker.from_env(timeout=60)
    started, container = time.monotonic(), None
    restore = ""
    if request.get("repository_metadata"):
        from swebench_io import restore_repository_metadata
        raw = Path(request["repository_metadata"]).read_bytes()
        expected = config.get("baseline_metadata_sha256", {}).get(request["task_id"])
        if expected and hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("Frozen repository metadata drift")
        (directory / "repository-metadata.json").write_bytes(raw)
        script = ("import json, os, posixpath\nfrom pathlib import Path\n" + inspect.getsource(restore_repository_metadata)
                  + "\nrestore_repository_metadata('/testbed', json.loads(Path('/judge/repository-metadata.json').read_text()))\n")
        (directory / "restore_metadata.py").write_text(script)
        restore = "python /judge/restore_metadata.py && "
    # Copy children, preserving their mtimes, without trying to chmod/utime the
    # root-owned tmpfs mount point itself. Include hidden source entries.
    command = (source_copy_command() + " && " + restore + restore_git_command(request["base_commit"]) + " && " +
               "git -C /testbed -c safe.directory=/testbed apply /judge/prediction.patch && "
               "echo 'CORECODER_PATCH_APPLIED' && /bin/bash /judge/eval.sh")
    log_path = directory / "test_output.txt"
    try:
        container = client.containers.run(
            request["image_id"], ["-c", command], entrypoint="/bin/bash", detach=True,
            name="corecoder-score-" + request["job_id"], user="65534:65534", working_dir="/testbed",
            network_disabled=True, read_only=True, mem_limit="4g", memswap_limit="4g", nano_cpus=2_000_000_000,
            pids_limit=256, cap_drop=["ALL"], security_opt=["no-new-privileges"],
            labels={"corecoder.eval.job": request["job_id"]},
            environment={"HOME": "/tmp", "MPLCONFIGDIR": "/tmp/mpl", "PYTHONDONTWRITEBYTECODE": "1", "PIP_USER": "1",
                         "PIP_NO_INDEX": "1", "PIP_FIND_LINKS": "/wheels"},
            log_config=docker.types.LogConfig(type="local", config={"max-size": "25m", "max-file": "2"}),
            # CPython must mmap native extensions from these disposable mounts.
            tmpfs={"/tmp": "rw,exec,nosuid,nodev,size=1g,mode=1777", "/testbed": "rw,exec,nosuid,nodev,size=3g,mode=1777"},
            volumes={request["base_directory"]: {"bind": "/source", "mode": "ro"}, str(directory): {"bind": "/judge", "mode": "ro"},
                     str(wheelhouse): {"bind": "/wheels", "mode": "ro"}})
        while True:
            container.reload()
            if container.status == "exited":
                break
            if time.monotonic() - started > request["score_seconds"]:
                raise TimeoutError("official_test_timeout")
            time.sleep(.5)
        logs = container.logs(stdout=True, stderr=True)
        if len(logs) > 50 * 2**20:
            raise ValueError("Official log exceeds 50 MiB cap")
        log_path.write_bytes(logs)
        if b"CORECODER_GIT_BASELINE_VERIFIED" not in logs:
            raise ValueError("Official scorer Git baseline unavailable; no score assigned")
        prediction = {"instance_id": row["instance_id"], "model_name_or_path": request.get("control", "corecoder"), "model_patch": patch}
        report = get_eval_report(spec, prediction, log_path, include_tests_status=True)
        (directory / "official-report.json").write_text(json.dumps(report, indent=2) + "\n")
        result = {"ok": True, "resolved": report[row["instance_id"]]["resolved"],
                  "git_baseline_verified": True,
                  "patch_applied": b"CORECODER_PATCH_APPLIED" in logs,
                  "seconds": time.monotonic() - started, "container_exit": container.attrs["State"]["ExitCode"],
                  "official_report": str(directory / "official-report.json")}
        (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))
    finally:
        if container:
            container.remove(force=True)
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", type=Path)
    parser.add_argument("--request", type=Path)
    args = parser.parse_args()
    if args.environment:
        environment(args.environment)
    else:
        score(json.loads(args.request.read_text()))
