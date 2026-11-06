"""Model-produced task data inside immutable operator-supplied permissions."""
import hashlib
import json
import shlex
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from corecoder_mini.delivery import read_json
from corecoder_mini.task_state import atomic_json
from corecoder_mini.tools.base import Tool


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)


class Requirement(StrictModel):
    id: str = Field(min_length=1, max_length=40)
    behavior: str = Field(min_length=1, max_length=600)
    source: Literal['user', 'inferred'] = 'user'


class Check(StrictModel):
    id: str = Field(min_length=1, max_length=40)
    command: str = Field(min_length=1, max_length=2000)
    seconds: int = Field(default=120, ge=1, le=180)


class Metric(StrictModel):
    id: str = Field(min_length=1, max_length=40)
    name: str = Field(min_length=1, max_length=100)
    kind: Literal['behavior', 'regression', 'performance', 'coverage', 'artifact']
    target: str = Field(min_length=1, max_length=300)
    requirement_ids: list[str] = Field(min_length=1, max_length=12)
    check_ids: list[str] = Field(default_factory=list, max_length=8)
    source: Literal['user', 'inferred'] = 'inferred'


class PlanStep(StrictModel):
    id: str = Field(min_length=1, max_length=40)
    content: str = Field(min_length=1, max_length=500)
    owner: Literal['main', 'child_1', 'child_2'] = 'main'
    dependencies: list[str] = Field(default_factory=list, max_length=12)
    requirements: list[str] = Field(min_length=1, max_length=12)
    write_scope: list[str] = Field(default_factory=list, max_length=12)
    evidence: str = Field(min_length=1, max_length=300)


def safe_relative(value):
    if not isinstance(value, str) or not value or '\\' in value or value.startswith('/') or ':' in value:
        raise ValueError('workspace_relative_posix_path_required')
    if any(p in {'', '..', '.git'} or p.startswith('.env') for p in value.split('/')):
        raise ValueError('unsafe_scope')
    return value


def in_scope(path, scopes):
    return any(scope == '.' or path == scope or path.startswith(scope.rstrip('/') + '/') for scope in scopes)


class Intent(StrictModel):
    kind: Literal['implement', 'fix', 'refactor', 'test', 'explain', 'locate', 'review']
    summary: str = Field(min_length=1, max_length=800)
    requirements: list[Requirement] = Field(min_length=1, max_length=12)
    constraints: list[str] = Field(default_factory=list, max_length=12)
    metrics: list[Metric] = Field(min_length=1, max_length=12)
    checks: list[Check] = Field(default_factory=list, max_length=8)
    plan: list[PlanStep] = Field(min_length=1, max_length=12)

    @model_validator(mode='after')
    def references(self):
        def unique(items):
            ids = [item.id for item in items]
            if len(ids) != len(set(ids)):
                raise ValueError('duplicate_id')
            return set(ids)
        requirements, checks = unique(self.requirements), unique(self.checks)
        unique(self.plan)
        unique(self.metrics)
        covered = set()
        for metric in self.metrics:
            if set(metric.requirement_ids) - requirements or set(metric.check_ids) - checks:
                raise ValueError('unknown_metric_reference')
            covered.update(metric.requirement_ids)
        if covered != requirements:
            raise ValueError('each_requirement_needs_a_metric')
        visited = set()
        for step in self.plan:
            if set(step.dependencies) - visited or set(step.requirements) - requirements:
                raise ValueError('plan_requires_ordered_acyclic_dependencies')
            for scope in step.write_scope:
                safe_relative(scope)
            visited.add(step.id)
        if self.kind in {'implement', 'fix', 'refactor', 'test'} and not self.checks:
            raise ValueError('mutating_task_requires_executable_public_checks')
        if self.kind in {'implement', 'fix', 'refactor', 'test'} and any(not m.check_ids for m in self.metrics):
            raise ValueError('each_metric_requires_an_executable_check')
        for check in self.checks:
            if 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT' in check.command:
                raise ValueError('check_cannot_submit')
        return self


def check_command(intent):
    # Public, model-proposed checks are not independent hidden correctness.
    return ' && '.join('timeout ' + str(c.seconds) + ' bash -lc ' + shlex.quote(c.command)
                       for c in intent.checks) or "python -c 'assert True'"


class TaskContractTool(Tool):
    name = 'task_contract'
    description = ('Convert the complete user request to one task JSON before editing: requirements, '
                   'inferred metrics, executable public checks and bounded dependency plan. '
                   'Workspace permissions come from the operator and cannot be expanded. '
                   'Metrics are proposals, not test results. Call once with the complete object.')
    parameters = Intent.model_json_schema()

    def __init__(self, state, environment, settings):
        self.state, self.environment, self.settings = state, environment, settings
        self.path = state.path.parent / 'task-contract.json'

    def validate_arguments(self, arguments):
        try:
            Intent.model_validate(arguments)
        except ValueError:
            raise ValueError('invalid_task_contract: check required fields, IDs, metrics and ordered dependencies') from None

    def execute(self, **arguments):
        intent = Intent.model_validate(arguments)
        if self.path.exists():
            previous = read_json(self.path)
            if previous['intent'] != intent.model_dump():
                raise ValueError('task_contract_is_frozen; use Todo/scratchpad for progress updates')
            return json.dumps({'stored': True, 'immutable': True, 'path': 'task-contract.json'})
        declared = self.settings.get('task_intent')
        if declared and declared != intent.kind:
            raise ValueError('intent_conflicts_with_operator_or_benchmark')
        scopes = self.settings.get('write_scope', ['.'])
        for step in intent.plan:
            if any(not in_scope(p, scopes) for p in step.write_scope):
                raise ValueError('plan_exceeds_operator_write_scope')
        operator = self.settings.get('operator_workspace', {})
        document = {'schema_version': 1, 'workspace': operator, 'allowed_write_scope': scopes,
                    'original_request_sha256': self.state.data.get('request_sha256'),
                    'intent': intent.model_dump(), 'metrics_are_proposed': True,
                    'official_evaluator_unchanged': self.settings.get('task_kind') == 'benchmark'}
        atomic_json(self.path, document)
        self.state.update(lambda d: d.update(intent={
            'kind': intent.kind, 'summary': intent.summary,
            'requirements': [{'id': r.id, 'behavior': r.behavior} for r in intent.requirements],
            'constraints': intent.constraints}, todos=[{
                'id': p.id, 'content': p.content, 'owner': p.owner, 'dependencies': p.dependencies,
                'requirements': p.requirements, 'status': 'pending'} for p in intent.plan]))
        if not self.environment.public_check_command and self.settings.get('contract_verification', True):
            self.environment.public_check_command = check_command(intent)
        return json.dumps({'stored': True, 'path': 'task-contract.json', 'requirements': len(intent.requirements),
                           'metrics': len(intent.metrics), 'steps': len(intent.plan),
                           'sha256': hashlib.sha256(self.path.read_bytes()).hexdigest(),
                           'next': 'Implement the scoped task, run the public checks, update Todo, then submit.'})


def install(state, environment, settings):
    tool = TaskContractTool(state, environment, settings)
    environment.registry.register(tool)
    environment.task_contract_path = tool.path
