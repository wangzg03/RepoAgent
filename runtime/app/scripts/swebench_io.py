"""Frozen cache integrity and filesystem patches; no host Git operations."""

import difflib
import hashlib
import json
import os
import posixpath
import shutil
import tarfile
from pathlib import Path


def repository_metadata(directory):
    directory = Path(directory)
    return directory.parent / (directory.name + ".metadata.json")


def extract_archive(archive, target, byte_cap):
    """Materialize in-tree file links on Windows; keep their topology for Linux.

    No archive link is followed during extraction. Targets must have been safely
    extracted as regular files; dangling, cyclic, directory and escaping links fail.
    """
    target = Path(target).resolve()
    target.mkdir(parents=True, exist_ok=False)
    total, pending, links, executables, seen = 0, [], {}, [], set()
    with tarfile.open(archive, "r|*") as source:
        for item in source:
            parts = item.name.split("/")
            if (parts[0] != "testbed" or any(p in {"..", "."} for p in parts)
                    or "\\" in item.name or ":" in item.name):
                raise ValueError("Unsafe repository archive path")
            name = "/".join(p for p in parts[1:] if p)
            if not name:
                continue
            if name.casefold() in seen:
                raise ValueError("Duplicate or case-colliding archive path")
            seen.add(name.casefold())
            dest = target / name
            if item.issym() or item.islnk():
                raw = item.linkname
                if not raw or raw.startswith("/") or "\\" in raw or ":" in raw:
                    raise ValueError("Unsafe archive link target")
                resolved = posixpath.normpath(posixpath.join(posixpath.dirname(name), raw) if item.issym()
                                              else raw.removeprefix("testbed/"))
                if resolved in {".", ".."} or resolved.startswith("../"):
                    raise ValueError("Archive link escaped repository root")
                pending.append((name, resolved))
                links[name] = {"target": raw, "kind": "symlink" if item.issym() else "hardlink"}
            elif item.isdir():
                dest.mkdir(parents=True, exist_ok=True)
            elif item.isfile():
                total += item.size
                if total > byte_cap:
                    raise ValueError("Repository export size limit")
                dest.parent.mkdir(parents=True, exist_ok=True)
                with dest.open("wb") as stream:
                    shutil.copyfileobj(source.extractfile(item), stream)
                os.utime(dest, (item.mtime, item.mtime))
                if item.mode & 0o111:
                    executables.append(name)
            else:
                raise ValueError("Unsupported repository archive member")
    while pending:
        unresolved = []
        for name, resolved in pending:
            original = target / resolved
            if not original.is_file() or original.is_symlink():
                unresolved.append((name, resolved))
                continue
            total += original.stat().st_size
            if total > byte_cap:
                raise ValueError("Repository export size limit")
            dest = target / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, dest)
        if len(unresolved) == len(pending):
            raise ValueError("Dangling, cyclic or directory archive links are unsupported")
        pending = unresolved
    repository_metadata(target).write_text(json.dumps({"schema": 1, "links": links,
        "executable_paths": executables}, indent=2) + "\n", encoding="utf-8")


def restore_repository_metadata(root, metadata):
    """Restore links only inside the disposable Linux scorer, never on the host."""
    root = Path(root).resolve()
    def path(name):
        if not isinstance(name, str) or not name or "\\" in name or ":" in name or name.startswith("/"):
            raise ValueError("Unsafe metadata path")
        value = root / name
        if not value.resolve().is_relative_to(root) or value == root or ".git" in value.parts:
            raise ValueError("Metadata path escaped repository")
        return value
    for name in metadata.get("executable_paths", []):
        path(name).chmod(0o755)
    for name, item in metadata.get("links", {}).items():
        dest = path(name)
        raw = item["target"]
        original = path(posixpath.normpath(posixpath.join(posixpath.dirname(name), raw)) if item["kind"] == "symlink"
                        else raw.removeprefix("testbed/"))
        if not original.is_file() or not dest.is_file():
            raise ValueError("Restored file link target unavailable")
        dest.unlink()
        if item["kind"] == "symlink":
            dest.symlink_to(raw)
        elif item["kind"] == "hardlink":
            os.link(original, dest)
        else:
            raise ValueError("Unsupported link metadata")


def tree_hashes(directory):
    root = Path(directory).resolve()
    if not root.is_dir():
        raise ValueError("Frozen baseline cache directory is missing")
    result = {}
    for p in root.rglob("*"):
        if set(p.relative_to(root).parts) & {".git", "__pycache__", ".pytest_cache"}:
            continue
        if p.is_symlink():
            raise ValueError("Baseline cache contains symbolic links")
        if p.is_file():
            result[p.relative_to(root).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return result


def filesystem_patch(baseline, workspace, limit=2 * 2**20):
    """Export actual text changes without invoking Git or touching its metadata."""
    roots = [Path(baseline).resolve(), Path(workspace).resolve()]
    skip = {".git", "__pycache__", ".pytest_cache"}
    files = []
    for root in roots:
        names = set()
        for path in root.rglob("*"):
            relative = path.relative_to(root)
            if set(relative.parts) & skip:
                continue
            if path.is_symlink():
                raise ValueError("Patch source contains an unsupported symbolic link")
            if path.is_file():
                names.add(relative.as_posix())
        files.append(names)
    chunks = []
    size = 0
    for name in sorted(files[0] | files[1]):
        if any(c in name for c in '\r\n\t"\\'):
            raise ValueError("Unsupported patch path")
        values = [(root / name).read_bytes() if name in names else None for root, names in zip(roots, files)]
        if values[0] == values[1]:
            continue
        if any(value is not None and b"\0" in value for value in values):
            raise ValueError("Binary modification cannot be silently omitted")
        text = [(value or b"").decode("utf-8").splitlines(keepends=True) for value in values]
        delta = list(difflib.unified_diff(*text, fromfile="a/" + name if values[0] is not None else "/dev/null",
                                        tofile="b/" + name if values[1] is not None else "/dev/null"))
        part = "diff --git a/" + name + " b/" + name + "\n"
        if values[0] is None:
            part += "new file mode 100644\n"
        elif values[1] is None:
            part += "deleted file mode 100644\n"
        if not delta:
            # Git's standard zero-byte file patch has mode/index headers and no
            # hunk. Preserve empty package markers instead of failing delivery.
            empty_blob = "e69de29"
            part += ("index 0000000.." + empty_blob + "\n" if values[0] is None else
                     "index " + empty_blob + "..0000000\n")
        for line in delta:
            part += line if line.endswith("\n") else line + "\n\\ No newline at end of file\n"
        size += len(part.encode())
        if size > limit:
            raise ValueError("Submitted patch exceeds 2 MiB")
        chunks.append(part)
    return "".join(chunks).encode()
