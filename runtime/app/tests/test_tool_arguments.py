"""Synthetic public Todo payloads over the actual parser and localhost SSE, no paid API/Docker."""

import json
from types import SimpleNamespace
from typing import ClassVar

import pytest
from test_native_upstream import local_provider, response

from corecoder_mini.budget import Budget
from corecoder_mini.events import Events
from corecoder_mini.native_model import NativeModel, NativeProvider
from corecoder_mini.runtime import local
from corecoder_mini.state_tools import TodoTool, WorkingState
from corecoder_mini.supervisor import DEFAULTS
from corecoder_mini.tools.arguments import decode_arguments
from corecoder_mini.tools.base import Tool
from corecoder_mini.tools.registry import ToolRegistry
from corecoder_mini.upstream import FormatError


class Probe(Tool):
    name = "bash"
    description = "Public fixture shell placeholder; never executed by parser tests."
    parameters: ClassVar[dict] = {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}

    def execute(self, command):
        raise AssertionError("Parser tests must not execute shell commands")


def todo():
    return {"intent": {"kind": "implement", "summary": "Build a public toy program",
                       "requirements": [{"id": "r1", "behavior": "Return a sum"}]},
            "tasks": [{"id": "t1", "content": "Implement sum", "status": "pending", "requirements": ["r1"]}]}


@pytest.fixture(autouse=True)
def clean_runtime():
    vars(local).clear()
    yield
    vars(local).clear()


def parser(tmp_path):
    model = NativeModel.__new__(NativeModel)
    model.directory, model.calls, model.events = tmp_path, 1, Events(tmp_path)
    model.registry = ToolRegistry([Probe(), TodoTool(WorkingState(tmp_path))])
    return model


def completed(arguments, name="todo_write", finish="tool_calls"):
    call = SimpleNamespace(id="fixture-call", function=SimpleNamespace(name=name, arguments=arguments))
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish, message=SimpleNamespace(tool_calls=[call]))])


@pytest.mark.parametrize("name", ["bash", "edit_file", "write_file", "memory", "subagent"])
def test_executable_and_other_tools_are_never_repaired(name):
    with pytest.raises(json.JSONDecodeError):
        decode_arguments(name, '{"value":1}}')


@pytest.mark.parametrize("raw", ['{"tasks":', '{"tasks":[]', '{"tasks":[]}}}',
                                 '{"tasks":[]}{}', '{"tasks":[]}ignore instructions', '[{}]}'])
def test_truncation_ambiguity_prose_and_arrays_rejected(raw):
    with pytest.raises(json.JSONDecodeError):
        decode_arguments("todo_write", raw)


def test_schema_validation_still_required(tmp_path):
    model = parser(tmp_path)
    with pytest.raises(FormatError):
        model._parse_actions(completed('{"tasks":"not-an-array"}}'))
    assert not (tmp_path / "private/argument-normalizations").exists()
    assert not model.registry.tools["todo_write"].state.data["todos"]


@pytest.mark.parametrize("finish", ["length", "content_filter", None])
def test_unfinished_or_rejected_response_cannot_recover(tmp_path, finish):
    model = parser(tmp_path)
    with pytest.raises(FormatError):
        model._parse_actions(completed(json.dumps(todo()) + "}", finish=finish))
    assert not (tmp_path / "private/argument-normalizations").exists()


def test_exact_repair_is_auditable_and_todo_executes_once(tmp_path):
    model, arguments = parser(tmp_path), todo()
    original = json.dumps(arguments) + "}"
    result = completed(original)
    actions = model._parse_actions(result)
    assert len(actions) == 1 and actions[0]["arguments"] == arguments
    assert not model.registry.tools["todo_write"].state.data["todos"]
    record = json.loads((tmp_path / "private/argument-normalizations/0001.json").read_text())
    assert record["repairs"][0]["original_arguments"] == original
    assert record["status"] == "validated_not_executed"
    assert json.loads(result.choices[0].message.tool_calls[0].function.arguments) == arguments
    model.registry.tools["todo_write"].execute(**actions[0]["arguments"])
    assert len(model.registry.tools["todo_write"].state.data["todos"]) == 1
    event = json.loads(next((tmp_path / "events").glob("*.json")).read_text())
    assert "original_arguments" not in json.dumps(event)


def test_mixed_batch_invalid_write_keeps_entire_batch_unexecuted(tmp_path):
    model = parser(tmp_path)
    result = completed(json.dumps(todo()) + "}")
    result.choices[0].message.tool_calls.append(
        SimpleNamespace(id="bad-call", function=SimpleNamespace(name="bash", arguments='{"command":"echo fixture"}}')))
    with pytest.raises(FormatError):
        model._parse_actions(result)
    assert not model.registry.tools["todo_write"].state.data["todos"]
    assert not (tmp_path / "private/argument-normalizations").exists()


@pytest.mark.parametrize("stream", [False, True])
def test_actual_local_transport_recovers_without_retry(tmp_path, monkeypatch, stream):
    monkeypatch.setenv("CORECODER_PUBLIC_FIXTURE_KEY", "sk-local-public-fixture")
    def reply(data, index):
        obj = response([("todo_write", todo()), ("bash", {"command": "echo public-fixture"})])
        obj["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] += "}"
        return 200, obj
    with local_provider(reply) as (url, requests):
        settings = {**DEFAULTS, "provider_stream": stream}
        provider = NativeProvider("fixture", url, "CORECODER_PUBLIC_FIXTURE_KEY",
                                  Budget(.1, token_limit=100000, request_limit=3), (1, 1), settings)
        registry = ToolRegistry([Probe(), TodoTool(WorkingState(tmp_path))])
        model = NativeModel(provider, tmp_path, registry, None, Events(tmp_path), {
            "format_error_template": "{{error}}", "observation_template": "{{output}}"})
        message = model.query([{"role": "user", "content": "Public fixture: record a Todo."}])
        assert len(requests) == 1
        assert provider.budget.requests == 1
        actions = message["extra"]["actions"]
        assert len(actions) == 2 and actions[0]["arguments"] == todo()
        registry.tools["todo_write"].execute(**actions[0]["arguments"])
        assert len(registry.tools["todo_write"].state.data["todos"]) == 1
        assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == todo()
