"""Per-execution state. Never change process cwd in a concurrent agent."""

import queue
import threading
import time
from pathlib import Path

local = threading.local()
WORKSPACE_MOUNT = "/workspace"


class Cancelled(RuntimeError):
    pass


class DeadlineExceeded(Cancelled):
    pass


def check_cancelled():
    deadline = getattr(local, "deadline", None)
    if deadline is not None and time.monotonic() >= deadline:
        event = getattr(local, "cancel", None)
        if event is not None:
            event.set()
        raise DeadlineExceeded("Task deadline exceeded; durable state can be resumed")
    event = getattr(local, "cancel", None)
    if event is not None and event.is_set():
        raise Cancelled("Task cancelled")


def remaining_time():
    check_cancelled()
    deadline = getattr(local, "deadline", None)
    return max(0.001, deadline - time.monotonic()) if deadline is not None else None


def interruptible_call(function, *, timeout=None):
    """Detach a blocked transport wait on cancellation; late results are disposed, never delivered."""
    event = getattr(local, "cancel", None)
    if event is None and timeout is None:
        return function()
    output = queue.Queue(maxsize=1)
    context = dict(vars(local))
    expires = time.monotonic() + timeout if timeout is not None else None
    abandoned, publish = threading.Event(), threading.Lock()

    def dispose(value):
        if hasattr(value, "close"):
            def close():
                try:
                    value.close()
                except (RuntimeError, ValueError, OSError):
                    pass
            threading.Thread(target=close, daemon=True).start()

    def run():
        vars(local).update(context)
        try:
            value = function()
            with publish:
                if abandoned.is_set() or event is not None and event.is_set():
                    dispose(value)
                else:
                    output.put((True, value))
        except BaseException as exc:  # noqa: BLE001 - transfer worker failure to owning thread
            with publish:
                if not abandoned.is_set():
                    output.put((False, exc))

    threading.Thread(target=run, daemon=True, name="corecoder-transport").start()
    delivered, pending = False, None
    try:
        while True:
            check_cancelled()
            if expires is not None and time.monotonic() >= expires:
                raise TimeoutError("Provider request deadline exceeded")
            try:
                success, value = output.get(timeout=0.05)
            except queue.Empty:
                continue
            pending = value if success else None
            check_cancelled()
            if expires is not None and time.monotonic() >= expires:
                raise TimeoutError("Provider request deadline exceeded")
            if success:
                delivered = True
                return value
            raise value
    finally:
        if not delivered:
            if pending is not None:
                dispose(pending)
            with publish:
                abandoned.set()
                try:
                    success, value = output.get_nowait()
                except queue.Empty:
                    pass
                else:
                    if success:
                        dispose(value)


def interruptible_stream(stream, *, timeout=None):
    event = getattr(local, "cancel", None)
    if event is None and timeout is None:
        yield from stream
        return
    output = queue.Queue(maxsize=8)
    context = dict(vars(local))
    stopped = threading.Event()
    expires = time.monotonic() + timeout if timeout is not None else None

    def stopping():
        return stopped.is_set() or event is not None and event.is_set()

    def put(item):
        while not stopping():
            try:
                output.put(item, timeout=0.05)
                return
            except queue.Full:
                continue

    def read():
        vars(local).update(context)
        try:
            for chunk in stream:
                if stopping():
                    break
                put(("chunk", chunk))
        except BaseException as exc:  # noqa: BLE001 - transfer worker failure to owning thread
            put(("error", exc))
        finally:
            if hasattr(stream, "close"):
                try:
                    stream.close()
                except (RuntimeError, ValueError, OSError):
                    pass
            put(("end", None))

    threading.Thread(target=read, daemon=True, name="corecoder-stream").start()
    completed = False
    try:
        while True:
            check_cancelled()
            if expires is not None and time.monotonic() >= expires:
                raise TimeoutError("Provider request deadline exceeded")
            try:
                kind, value = output.get(timeout=0.05)
            except queue.Empty:
                continue
            if kind == "end":
                completed = True
                return
            if kind == "error":
                raise value
            yield value
    finally:
        stopped.set()
        if not completed and hasattr(stream, "close"):
            # Closing a provider socket may itself block; do not hold up logical cancellation.
            def close():
                try:
                    stream.close()
                except (RuntimeError, ValueError, OSError):
                    pass
            threading.Thread(target=close, daemon=True).start()


def retry_delay(seconds):
    import time
    event = getattr(local, "cancel", None)
    if event is None:
        time.sleep(seconds)
    else:
        event.wait(seconds)
        check_cancelled()


def workspace_path(value, root, *, cwd=None, mount=None, write=False):
    """Resolve a configured mount alias, then enforce the actual task boundary.

    The mount comes from the execution backend, never from model arguments.
    Resolve symlinks before granting access. Existing native absolute paths keep
    their meaning even when a Linux workspace itself lives below /workspace.
    """
    root = Path(root).resolve()
    path = Path(value).expanduser()
    spelling = str(value).replace("\\", "/")
    native_inside = path.is_absolute() and path.is_relative_to(root)
    if not native_inside and mount and (spelling == mount or spelling.startswith(mount + "/")):
        path = root / spelling[len(mount):].lstrip("/")
    else:
        # On Windows '/other' has an anchor but is_absolute() is false. It must
        # not silently become relative to the task's drive or current directory.
        if path.anchor and not path.is_absolute():
            raise ValueError("Path outside task workspace; use workspace-relative paths ('.' is the task root)")
        if not path.is_absolute():
            path = Path(cwd or root) / path
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("Path outside task workspace; use workspace-relative paths ('.' is the task root)")
    # Include Windows trailing-dot/space aliases and named streams of .git.
    if write and any(part.split(":", 1)[0].rstrip(" .").casefold() == ".git"
                     for part in (*path.parts, *resolved.relative_to(root).parts)):
        raise ValueError("Writes to git metadata are forbidden")
    return resolved


def resolve_path(value: str, *, write=False) -> Path:
    root = getattr(local, "root", None)
    if root is not None:
        return workspace_path(value, root, cwd=getattr(local, "cwd", None),
                              mount=getattr(local, "workspace_mount", None), write=write)
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(getattr(local, "cwd", None) or Path.cwd()) / path
    return path.resolve()


def display_path(path):
    root = getattr(local, "root", None)
    if root is not None:
        return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    return str(path)


def missing_path(path):
    return (f"Error: {display_path(path)} not found. "
            "Use workspace-relative paths; '.' is already the repository root. "
            "Search by filename using glob or code_search from '.'; do not guess extra root directories.")
