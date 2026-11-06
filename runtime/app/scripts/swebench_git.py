"""Private original-image Git context for the isolated official scorer only."""
import hashlib
import subprocess
import tarfile
import time
import uuid
from pathlib import Path


def validate_git_archive(path, limit=1024**3):
    total = 0
    with tarfile.open(path, "r:gz") as archive:
        for item in archive:
            parts = item.name.split("/")
            if (parts[0] != ".git" or any(p in {"..", "."} for p in parts) or "\\" in item.name
                    or ":" in item.name or not (item.isdir() or item.isfile())):
                raise ValueError("Unsafe private Git archive member")
            total += item.size
            if total > limit:
                raise ValueError("Private Git archive exceeds size limit")
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_git(image, commit, archive, job_id, timeout=300):
    """Read a pinned public image; never invoke host/project Git or expose it to inference."""
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("Invalid original base commit")
    archive = Path(archive)
    archive.parent.mkdir(parents=True, exist_ok=True)
    name = "corecoder-git-" + uuid.uuid4().hex
    command = ("cp -a --no-preserve=ownership /testbed /tmp/repository && "
               "git -C /tmp/repository -c safe.directory=/tmp/repository -c core.hooksPath=/tmp/no-hooks "
               "checkout --detach --force " + commit + " >&2 && "
               "test \"$(git -C /tmp/repository -c safe.directory=/tmp/repository rev-parse HEAD)\" = " + commit + " && "
               "tar --exclude=.git/hooks --exclude=.git/logs -C /tmp/repository -czf - .git")
    argv = ["docker", "run", "--rm", "--pull=never", "--name", name, "--label", "corecoder.eval.job=" + job_id,
            "--network=none", "--read-only", "--user=65534:65534", "--cpus=2", "--memory=4g", "--memory-swap=4g",
            "--pids-limit=128", "--cap-drop=ALL", "--security-opt=no-new-privileges",
            "--tmpfs=/tmp:rw,exec,nosuid,nodev,size=3g,mode=1777", "--entrypoint=/bin/bash", image, "-c", command]
    with archive.open("xb") as output, archive.with_suffix(".stderr.log").open("xb") as error:
        process = subprocess.Popen(argv, stdout=output, stderr=error)
        started = time.monotonic()
        try:
            while process.poll() is None:
                if time.monotonic() - started > timeout or archive.stat().st_size > 512 * 1024**2:
                    raise TimeoutError("Private Git export deadline/size exceeded")
                time.sleep(.2)
            if process.returncode:
                raise RuntimeError("Pinned-image private Git export failed")
        finally:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=30, check=False)
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=15)
    return validate_git_archive(archive)


def restore_git_command(commit=None):
    if commit is not None and (len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit)):
        raise ValueError("Invalid original base commit")
    head_check = ("test \"$(git -C /testbed rev-parse HEAD)\" = " + commit + " && ") if commit else ""
    return ("tar --no-same-owner --no-same-permissions -xzf /judge/repository-git.tar.gz -C /testbed && "
            "git config --global --add safe.directory /testbed && "
            "git -C /testbed config core.hooksPath /tmp/corecoder-no-hooks && "
            "git -C /testbed rev-parse --is-inside-work-tree && "
            + head_check + "git -C /testbed diff --quiet HEAD -- . && echo CORECODER_GIT_BASELINE_VERIFIED")
