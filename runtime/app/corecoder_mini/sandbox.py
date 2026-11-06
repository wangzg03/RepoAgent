"""Fail-closed Linux container execution. The Docker daemon is trusted."""

import os
import subprocess
import threading
import time
import uuid
from pathlib import Path

from .runtime import WORKSPACE_MOUNT, check_cancelled


class DockerSandbox:
    def __init__(self, root, image="python:3.12-slim", read_only=False):
        self.root = Path(root).resolve(strict=True)
        self.image = image
        self.read_only = read_only
        if not self.root.is_dir() or self.root == Path(self.root.anchor):
            raise ValueError("Sandbox root must be a workspace directory, never a drive root")
        if "," in str(self.root):
            raise ValueError("Docker mount paths containing commas are unsupported")

    def command(self, command, cwd, name, timeout):
        relative = Path(cwd).resolve().relative_to(self.root)
        mount = f"type=bind,src={self.root},dst={WORKSPACE_MOUNT}"
        if self.read_only:
            mount += ",readonly"
        user = f"{os.getuid()}:{os.getgid()}" if hasattr(os, "getuid") and os.getuid() else "65534:65534"
        git_path = self.root / ".git"
        git_mount = (["--mount", f"type=bind,src={git_path},dst={WORKSPACE_MOUNT}/.git,readonly"]
                     if git_path.exists() and not git_path.is_symlink() else [])
        return [
            "docker",
            "run",
            "--rm",
            "--pull=never",
            "--name",
            name,
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--pids-limit=128",
            "--memory=512m",
            "--memory-swap=512m",
            "--cpus=1",
            "--ulimit",
            "nofile=256:256",
            "--user",
            user,
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=64m,mode=1777",
            "--mount",
            mount,
            *git_mount,
            "--workdir",
            WORKSPACE_MOUNT + "/" + relative.as_posix(),
            "--env",
            "HOME=/tmp",
            "--stop-timeout=1",
            "--entrypoint",
            "/bin/sh",
            self.image,
            "-c",
            command,
        ]

    def run(self, command, cwd, timeout=120, input_text=None):
        if not isinstance(timeout, int) or not 1 <= timeout <= 3600:
            raise ValueError("timeout must be between 1 and 3600 seconds")
        name = "corecoder-" + uuid.uuid4().hex
        argv = self.command(command, cwd, name, timeout)
        if input_text is not None:
            if not isinstance(input_text, str):
                raise TypeError("Sandbox stdin must be text")
            argv.insert(argv.index(self.image), "-i")
        # Drain both pipes with bounded head/tail buffers, even for infinite output.
        proc = subprocess.Popen(
            argv, stdin=subprocess.PIPE if input_text is not None else None, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        buffers = [[b"", b"", 0], [b"", b"", 0]]

        def feed_input():
            try:
                proc.stdin.write(input_text.encode("utf-8"))
            except (BrokenPipeError, OSError):
                pass
            finally:
                try:
                    proc.stdin.close()
                except (BrokenPipeError, OSError):
                    pass

        writer = threading.Thread(target=feed_input, daemon=True) if input_text is not None else None
        if writer:
            writer.start()

        def drain(pipe, buffer):
            try:
                while True:
                    part = pipe.read(4096)
                    if not part:
                        break
                    buffer[2] += len(part)
                    missing = max(0, 12000 - len(buffer[0]))
                    buffer[0] += part[:missing]
                    buffer[1] = (buffer[1] + part[missing:])[-6000:]
            finally:
                pipe.close()

        readers = [
            threading.Thread(target=drain, args=(pipe, buffer), daemon=True)
            for pipe, buffer in zip((proc.stdout, proc.stderr), buffers)
        ]
        for reader in readers:
            reader.start()
        try:
            deadline = time.monotonic() + timeout
            while proc.poll() is None:
                check_cancelled()
                if time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired(argv, timeout)
                time.sleep(0.05)
            for reader in readers:
                reader.join(timeout=5)

            def capture(buffer):
                head, tail, length = buffer
                marker = b"\n... output truncated ...\n" if length > 18000 else b""
                return (head + marker + tail).decode("utf-8", errors="replace")

            return subprocess.CompletedProcess(argv, proc.returncode, capture(buffers[0]), capture(buffers[1]))
        finally:
            # Killing the docker CLI alone leaves the container running.
            try:
                subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=10, check=False)
            finally:
                if proc.poll() is None:
                    proc.kill()
                proc.wait(timeout=10)
                for reader in readers:
                    reader.join(timeout=5)
                if writer:
                    writer.join(timeout=5)
