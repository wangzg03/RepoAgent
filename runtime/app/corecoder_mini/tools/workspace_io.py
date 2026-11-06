"""Stdlib file worker executed with python -I inside the task's Linux sandbox.

Exact unique anchors follow the original CoreCoder editor and pi's edit contract.
Paths and all file operations are resolved inside /workspace using directory FDs.
"""

import difflib
import hashlib
import json
import os
import re
import stat
import sys
import uuid

MAX_FILE_BYTES = 2 * 2**20


class FileToolError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def validate_path(value):
    if not isinstance(value, str) or not 1 <= len(value) <= 512 or any(
            ord(c) < 32 or c in '\\:"' for c in value):
        raise FileToolError('unsafe_path', 'Use a workspace-relative POSIX file path, such as src/module.py.')
    value = value.removeprefix('/workspace/')
    parts = value.split('/')
    if any(p in {'', '.', '..'} or p != p.rstrip(' .') or p.casefold() in {'.git', '.corecoder', '.pytest_cache', 'key.txt'}
           or p.casefold().startswith('.env') for p in parts):
        raise FileToolError('unsafe_path', 'Path must stay in the workspace and avoid credentials, metadata and tool cache.')
    return parts


def validate_arguments(operation, arguments):
    validate_path(arguments['file_path'])
    expected = arguments.get('expected_sha256')
    if expected is not None and (not isinstance(expected, str) or not re.fullmatch('[0-9a-f]{64}', expected)):
        raise FileToolError('invalid_hash', 'expected_sha256 must be the 64 lowercase hex characters from read_file.')
    if operation == 'edit_file':
        old, new = arguments['old_string'], arguments['new_string']
        if not isinstance(old, str) or not old or len(old) > 32768 or not isinstance(new, str) or len(new) > 65536:
            raise FileToolError('invalid_edit', 'Provide a nonempty exact old_string and a bounded new_string.')
        if old == new:
            raise FileToolError('no_change', 'old_string equals new_string; provide an actual change.')
        if '\0' in old or '\0' in new:
            raise FileToolError('invalid_edit', 'Code edits must be UTF-8 text without NUL bytes.')
    if operation == 'write_file' and (not isinstance(arguments['content'], str) or len(arguments['content']) > 65536):
        raise FileToolError('invalid_content', 'content must be UTF-8 text with at most 65536 characters.')
    if operation == 'write_file' and '\0' in arguments['content']:
        raise FileToolError('invalid_content', 'Code files must be UTF-8 text without NUL bytes.')
    if operation == 'read_file':
        for key, default, maximum in [('offset', 1, 10**9), ('column', 1, 10**9), ('limit', 160, 1000)]:
            value = arguments.get(key, default)
            if type(value) is not int or not 1 <= value <= maximum:
                raise FileToolError('invalid_range', 'offset/column must be positive; limit must be 1..1000.')


def parent_fd(parts, create=False):
    fd = os.open('/workspace', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in parts[:-1]:
            if create:
                try:
                    os.mkdir(part, mode=0o755, dir_fd=fd)
                except FileExistsError:
                    pass
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        return fd, parts[-1]
    except BaseException:
        os.close(fd)
        raise


def read_bytes(fd, name):
    try:
        source = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
    except FileNotFoundError:
        return None, None
    with os.fdopen(source, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_FILE_BYTES:
            raise FileToolError('unsupported_file', 'Only regular UTF-8 text files up to 2 MiB are supported.')
        data = stream.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES or b'\0' in data:
        raise FileToolError('unsupported_file', 'Binary or oversized files are not supported.')
    data.decode('utf-8-sig')  # Reject invalid encoding before any mutation.
    return data, info


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_window(data, arguments):
    lines = data.decode('utf-8-sig').splitlines(keepends=True)
    offset, column, limit = arguments.get('offset', 1), arguments.get('column', 1), arguments.get('limit', 160)
    if (lines and offset > len(lines)) or (not lines and (offset != 1 or column != 1)):
        raise FileToolError('invalid_range', 'offset exceeds the file; read from an existing line.')
    if lines and column > len(lines[offset - 1]) + 1:
        raise FileToolError('invalid_range', 'column exceeds the selected line.')
    row, col, budget, fragments = offset - 1, column - 1, 4500, []
    while row < min(len(lines), offset - 1 + limit):
        segment = lines[row][col:]
        low, high = 0, min(len(segment), budget)
        while low < high:
            middle = (low + high + 1) // 2
            if len(json.dumps(segment[:middle], ensure_ascii=False)) - 2 <= budget:
                low = middle
            else:
                high = middle - 1
        piece = segment[:low]
        fragments.append(piece)
        budget -= len(json.dumps(piece, ensure_ascii=False)) - 2
        col += len(piece)
        if col < len(lines[row]):
            break
        row, col = row + 1, 0
    return {'sha256': sha256(data), 'size_bytes': len(data), 'total_lines': len(lines),
            'offset': offset, 'column': column, 'content': ''.join(fragments),
            'next_offset': row + 1 if row < len(lines) else None,
            'next_column': col + 1 if row < len(lines) else None}


def exact_anchor_help(file_path, content, old_string, new_string):
    """Reuse the original editor's exact sub-edit suggestions, never fuzzy writes."""
    if max(len(old_string), len(new_string)) > 12000 or len(content) > 2_000_000:
        return 'Read a targeted read_file window and retry using its exact text.'
    old, new = old_string.splitlines(), new_string.splitlines()
    suggestions = []
    for tag, i, j, start, end in difflib.SequenceMatcher(None, old, new).get_opcodes():
        if tag == 'replace' and 0 < j - i <= 3 and 0 < end - start <= 3:
            anchor, replacement = '\n'.join(old[i:j]), '\n'.join(new[start:end])
            if anchor.strip() and len(anchor) + len(replacement) <= 800 and content.count(anchor) == 1:
                suggestions.append({'file_path': file_path, 'old_string': anchor, 'new_string': replacement})
    if suggestions:
        return 'Review and resubmit one exact changed-line suggestion: ' + json.dumps(suggestions[:3], ensure_ascii=False)
    return 'No changes made. Use read_file to obtain a smaller unique exact anchor, without line numbers.'


def encoded_replacement(text, original):
    if b'\r\n' in original and b'\n' not in original.replace(b'\r\n', b''):
        text = text.replace('\r\n', '\n').replace('\n', '\r\n')
    return text.encode('utf-8')


def atomic_write(fd, name, data, original, info):
    # Staging shares the workspace filesystem, so rename is atomic. The already
    # excluded cache keeps a killed pre-commit temporary file out of the patch.
    staging, _ = parent_fd(['.pytest_cache', 'corecoder-filetools', 'unused'], create=True)
    temporary = uuid.uuid4().hex + '.tmp'
    try:
        output = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=staging)
        with os.fdopen(output, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fchmod(stream.fileno(), stat.S_IMODE(info.st_mode) if info else 0o644)
            os.fsync(stream.fileno())
        current, _ = read_bytes(fd, name)
        if current != original:
            raise FileToolError('version_conflict', 'File changed during the edit. Read the current version and retry; no overwrite performed.')
        if original is None:
            # Exclusive creation: a concurrently created file must never be replaced.
            try:
                os.link(temporary, name, src_dir_fd=staging, dst_dir_fd=fd, follow_symlinks=False)
            except FileExistsError:
                raise FileToolError('version_conflict', 'File was created concurrently. Read it before retrying; no overwrite performed.') from None
        else:
            os.replace(temporary, name, src_dir_fd=staging, dst_dir_fd=fd)
    finally:
        try:
            os.unlink(temporary, dir_fd=staging)
        except FileNotFoundError:
            pass
        os.close(staging)


def perform(operation, arguments):
    validate_arguments(operation, arguments)
    parts = validate_path(arguments['file_path'])
    fd, name = parent_fd(parts, create=operation == 'write_file')
    try:
        original, info = read_bytes(fd, name)
        file_path = '/'.join(parts)
        if operation == 'read_file':
            if original is None:
                raise FileToolError('file_not_found', 'File not found; search the workspace for the correct relative path.')
            return {'file_path': file_path, **read_window(original, arguments)}
        expected = arguments.get('expected_sha256')
        if expected is not None and (original is None or sha256(original) != expected):
            raise FileToolError('version_conflict', 'expected_sha256 does not match. Read the current file before retrying; no changes made.')
        if operation == 'edit_file':
            if original is None:
                raise FileToolError('file_not_found', 'Use write_file to create a new file; edit_file requires an existing file.')
            old = encoded_replacement(arguments['old_string'], original)
            first = original.find(old)
            count = 0 if first < 0 else (2 if original.find(old, first + 1) >= 0 else 1)
            if count == 0:
                raise FileToolError('anchor_not_found', exact_anchor_help(file_path, original.decode('utf-8-sig'),
                                                                        arguments['old_string'], arguments['new_string']))
            if count != 1:
                raise FileToolError('ambiguous_anchor', 'old_string matches at least twice, including overlaps. Include enough surrounding text to select exactly one location.')
            updated = original.replace(old, encoded_replacement(arguments['new_string'], original), 1)
        elif operation == 'write_file':
            if original is not None and expected is None:
                raise FileToolError('expected_hash_required', 'Overwriting requires expected_sha256 from read_file. Prefer edit_file for a targeted change.')
            updated = encoded_replacement(arguments['content'], original or b'')
            if original and original.startswith(b'\xef\xbb\xbf') and not updated.startswith(b'\xef\xbb\xbf'):
                updated = b'\xef\xbb\xbf' + updated
        else:
            raise FileToolError('unknown_operation', 'Unknown file operation.')
        if updated == original:
            raise FileToolError('no_change', 'The requested operation produces no change.')
        if len(updated) > MAX_FILE_BYTES:
            raise FileToolError('unsupported_file', 'Updated file exceeds 2 MiB.')
        diff = ''.join(difflib.unified_diff((original or b'').decode('utf-8-sig').splitlines(True),
                                          updated.decode('utf-8-sig').splitlines(True),
                                          fromfile='a/' + file_path, tofile='b/' + file_path))
        receipt = {'file_path': file_path, 'changed': True, 'created': original is None,
                   'before_sha256': sha256(original) if original is not None else None,
                   'after_sha256': sha256(updated), 'size_bytes': len(updated),
                   'diff': diff[:3000], 'diff_truncated': len(diff) > 3000}
        atomic_write(fd, name, updated, original, info)
        return receipt
    finally:
        os.close(fd)


def main():
    try:
        payload = json.loads(sys.stdin.read(2**20 + 1))
        result = perform(payload['operation'], payload['arguments'])
    except (ValueError, KeyError, OSError, UnicodeError) as exc:
        result = {'error': getattr(exc, 'code', 'file_io_error'),
                  'message': str(exc)[:2000] if isinstance(exc, FileToolError) else
                  'File access failed. Check the relative path, regular-file type, UTF-8 encoding and sandbox permissions.'}
        print(json.dumps(result, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
