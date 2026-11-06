"""Structured test execution against the current artifact in a disposable copy."""
import shutil
import uuid
from pathlib import Path
from typing import ClassVar

from corecoder_mini.engine import ScopedDocker
from corecoder_mini.task_state import atomic_json
from corecoder_mini.tools.base import Tool


class RunTestsTool(Tool):
    name = 'run_tests'
    description = ('Run an actual test/assertion command in a disposable Docker workspace copy. '
                   'Returns command, returncode and source snapshot hash. Does not modify main. '
                   'Use for scoped tasks, regression and public checks; failed checks require repair.')
    parameters: ClassVar[dict] = {'type': 'object', 'properties': {
        'command': {'type': 'string', 'minLength': 1, 'maxLength': 8000},
        'seconds': {'type': 'integer', 'minimum': 1, 'maximum': 180}},
        'required': ['command'], 'additionalProperties': False}

    def __init__(self, environment, directory):
        self.environment, self.directory = environment, Path(directory)

    def execute(self, command, seconds=120):
        from corecoder_mini.tools.base import FILE_MUTATION_LOCK
        from scripts.swebench_io import tree_hashes
        sandbox = self.environment.registry.tools['bash'].sandbox
        target = self.directory / 'test-workspaces' / uuid.uuid4().hex
        with FILE_MUTATION_LOCK:
            before = tree_hashes(sandbox.root)
            shutil.copytree(sandbox.root, target, ignore=shutil.ignore_patterns('.git', '__pycache__', '.pytest_cache'))
        hashes = tree_hashes(target)
        if hashes != before:
            raise ValueError('source_changed_during_test_snapshot')
        runner = ScopedDocker(target, sandbox.image, sandbox.job_id)
        observed = runner.run(command, target, seconds)
        if tree_hashes(sandbox.root) != before:
            return {'returncode': 125, 'output': 'Source changed while tests ran; result is stale. Rerun against current code.',
                    'exception_info': 'stale_test_snapshot'}
        output = observed.stdout + observed.stderr
        receipt = {'command': command, 'returncode': observed.returncode, 'output': output,
                   'source_sha256': hashes, 'isolated_workspace': str(target)}
        atomic_json(target.parent / (target.name + '.json'), receipt)
        return {'returncode': observed.returncode, 'output': output[:18000],
                'test_receipt': str(target.parent / (target.name + '.json')),
                'exception_info': '' if observed.returncode == 0 else 'test_failed'}
