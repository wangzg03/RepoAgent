"""Structured file operations through the existing cancellable Docker boundary."""

import json
import shlex
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import ClassVar

from ..runtime import check_cancelled
from .base import FILE_MUTATION_LOCK, Tool
from .workspace_io import validate_arguments

PATH = {'type': 'string', 'minLength': 1, 'maxLength': 512,
        'description': 'Workspace-relative POSIX file path, e.g. src/module.py.'}
HASH = {'type': 'string', 'minLength': 64, 'maxLength': 64,
        'description': 'Optional current file SHA256 returned by read_file; required to overwrite with write_file.'}


@contextmanager
def mutation_lock():
    while not FILE_MUTATION_LOCK.acquire(timeout=.05):
        check_cancelled()
    try:
        check_cancelled()
        yield
    finally:
        FILE_MUTATION_LOCK.release()


class WorkspaceFileTool(Tool):
    def __init__(self, sandbox, timeout, prefix=''):
        self.sandbox, self.timeout, self.prefix = sandbox, timeout, prefix
        self.program = Path(__file__).with_name('workspace_io.py').read_text(encoding='utf-8')

    def validate_arguments(self, arguments):
        validate_arguments(self.name, arguments)

    def _execute(self, arguments):
        check_cancelled()
        if not self.read_only and self.sandbox.read_only:
            raise ValueError('Read-only child cannot edit or create files.')
        self.validate_arguments(arguments)
        command = self.prefix + 'python -I -c ' + shlex.quote(self.program)
        options = {'submission_requested': False} if getattr(self.sandbox, 'native_submission', False) else {}
        try:
            proc = self.sandbox.run(command, self.sandbox.root, self.timeout,
                                    input_text=json.dumps({'operation': self.name, 'arguments': arguments}, ensure_ascii=False), **options)
        except subprocess.TimeoutExpired:
            return {'returncode': 124, 'output': 'File operation exceeded its deadline; inspect the current file before retrying.',
                    'exception_info': 'tool_timeout'}
        return {'returncode': proc.returncode, 'output': proc.stdout + proc.stderr,
                'exception_info': '' if proc.returncode == 0 else 'file_operation_failed'}


class ReadFileTool(WorkspaceFileTool):
    name = 'read_file'
    description = 'Read a bounded UTF-8 source window and its SHA256. Follow next_offset/next_column to continue. Content contains exact text without added line numbers.'
    read_only = concurrency_safe = True
    parameters: ClassVar[dict] = {'type': 'object', 'properties': {'file_path': PATH,
        'offset': {'type': 'integer', 'minimum': 1}, 'column': {'type': 'integer', 'minimum': 1},
        'limit': {'type': 'integer', 'minimum': 1, 'maximum': 1000}},
        'required': ['file_path'], 'additionalProperties': False}

    def execute(self, file_path, offset=1, limit=160, column=1):
        return self._execute({'file_path': file_path, 'offset': offset, 'limit': limit, 'column': column})


class EditFileTool(WorkspaceFileTool):
    name = 'edit_file'
    description = 'Modify existing code with one exact unique old_string -> new_string replacement. Use a small unique source anchor. Returns the actual diff and hashes. Ambiguous/stale matches do not write; correct the anchor using read_file.'
    parameters: ClassVar[dict] = {'type': 'object', 'properties': {'file_path': PATH,
        'old_string': {'type': 'string', 'minLength': 1, 'maxLength': 32768},
        'new_string': {'type': 'string', 'maxLength': 65536}, 'expected_sha256': HASH},
        'required': ['file_path', 'old_string', 'new_string'], 'additionalProperties': False}

    def execute(self, file_path, old_string, new_string, expected_sha256=None):
        arguments = {'file_path': file_path, 'old_string': old_string, 'new_string': new_string}
        if expected_sha256 is not None:
            arguments['expected_sha256'] = expected_sha256
        with mutation_lock():
            return self._execute(arguments)


class WriteFileTool(WorkspaceFileTool):
    name = 'write_file'
    description = 'Create a UTF-8 file and missing parent directories. Overwriting an existing file requires its current expected_sha256 from read_file. Prefer edit_file for a small code change. Returns actual hashes and a bounded diff.'
    parameters: ClassVar[dict] = {'type': 'object', 'properties': {'file_path': PATH,
        'content': {'type': 'string', 'maxLength': 65536}, 'expected_sha256': HASH},
        'required': ['file_path', 'content'], 'additionalProperties': False}

    def execute(self, file_path, content, expected_sha256=None):
        arguments = {'file_path': file_path, 'content': content}
        if expected_sha256 is not None:
            arguments['expected_sha256'] = expected_sha256
        with mutation_lock():
            return self._execute(arguments)
