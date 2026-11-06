"""Pinned DockerEnvironment lifecycle/execute, with bounded output and isolation."""
# isort: skip_file

import shlex
import subprocess
import uuid
from pathlib import Path

from . import upstream as _upstream  # noqa: F401
from minisweagent.environments.docker import DockerEnvironment

from .engine import SUBMIT, ScopedDocker
from .runtime import Cancelled, check_cancelled, interruptible_call


class IsolatedDockerEnvironment(DockerEnvironment):
    def _check_finished(self, output):
        # Do not submit because a source file happens to print the marker.
        if self.submission_requested:
            return super()._check_finished(output)

    def cleanup(self):
        # Upstream cleanup uses a POSIX background shell; use an argv on Windows.
        identity = getattr(self, "container_id", None)
        if identity:
            self.container_id = None
            subprocess.run([self.config.executable, "rm", "-f", identity], capture_output=True, timeout=15, check=False)


class SessionDocker(ScopedDocker):
    """One container per agent. Files and /tmp survive successive commands."""

    native_submission = True

    def __init__(self, root, image, job_id, read_only=False, owner_job_id=None):
        super().__init__(root, image, job_id, read_only)
        self.owner_job_id = owner_job_id or job_id
        self.environment = None

    def _ensure(self):
        if self.environment:
            return
        git_path = self.root / ".git"
        if git_path.is_symlink():
            raise ValueError("sandbox_git_metadata_symlink")
        # An empty protected mount prevents creating real metadata when the
        # input copy intentionally excludes Git. Never invoke project Git.
        git_path.mkdir(exist_ok=True) if not git_path.exists() else None
        argv = self.command("", self.root, "unused-name", 30)
        args = argv[2:argv.index("--entrypoint")]
        index = args.index("--name")
        del args[index:index + 2]
        label = args.index("corecoder.eval.job=" + self.job_id)
        args[label] = "corecoder.eval.job=" + self.owner_job_id
        args += ["--label", "corecoder.eval.agent=" + self.job_id]
        self.environment = IsolatedDockerEnvironment(
            image=self.image, cwd="/workspace", timeout=180, pull_timeout=30,
            container_timeout="2h", run_args=args,
            # Copied source excludes bytecode. Prevent timestamp/size-based stale
            # imports when a fast edit and recheck occur within the same second.
            env={"HOME": "/tmp", "PYTHONDONTWRITEBYTECODE": "1"}, forward_env=[])
        self.environment.submission_requested = False

    def run(self, command, cwd, timeout=120, input_text=None, *, submission_requested=None):
        check_cancelled()
        if type(timeout) is not int or not 1 <= timeout <= 3600:
            raise ValueError("timeout must be between 1 and 3600 seconds")
        self._ensure()
        relative = Path(cwd).resolve().relative_to(self.root).as_posix()
        wrapped = command
        if input_text is not None:
            marker = "CORECODER_STDIN_" + uuid.uuid4().hex
            wrapped = f"{command} <<'{marker}'\n{input_text}\n{marker}"
        # The upstream executor captures stdout. Spool inside its bounded /tmp
        # so arbitrary command output cannot grow the host's capture buffer.
        # GNU timeout owns this command's process group, not unrelated processes.
        wrapped = ("set +e; log=$(mktemp /tmp/corecoder-output.XXXXXX) || exit 125; "
                   f"timeout --kill-after=1 {timeout} bash -c {shlex.quote(wrapped)} >\"$log\" 2>&1; rc=$?; "
                   "size=$(wc -c <\"$log\"); if [ \"$size\" -le 18000 ]; then cat \"$log\"; "
                   "else head -c 12000 \"$log\"; printf '\\n... captured output bounded ...\\n'; "
                   "tail -c 6000 \"$log\"; fi; rm -f \"$log\"; exit \"$rc\"")
        self.environment.submission_requested = command.strip() == SUBMIT if submission_requested is None else submission_requested
        try:
            result = interruptible_call(lambda: self.environment.execute(
                {"command": wrapped}, cwd="/workspace/" + relative, timeout=timeout + 5))
        except Cancelled:
            self.cleanup()
            raise
        return subprocess.CompletedProcess([], result["returncode"], result["output"], "")

    def cleanup(self):
        if self.environment:
            self.environment.cleanup()
            self.environment = None
