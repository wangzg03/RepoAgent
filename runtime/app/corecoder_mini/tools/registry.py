"""Validated tool catalog. Extensions use the same contract as built-in tools."""

import inspect
import math
import re
from importlib.metadata import entry_points

from .base import Tool


def validate_value(value, schema, path="arguments"):
    """Validate the JSON-schema subset used by tool declarations, without coercion."""
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path} must be one of {schema['enum']}")
    kind = schema.get("type")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} must be finite")
    kinds = kind if isinstance(kind, list) else [kind]
    types = {"object": dict, "array": list, "string": str, "boolean": bool, "integer": int,
             "number": (int, float), "null": type(None)}
    if kind and not any(k in types and isinstance(value, types[k]) and
                        not (k in {"integer", "number"} and isinstance(value, bool)) for k in kinds):
        raise ValueError(f"{path} must have type {kind}")
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                raise ValueError(f"{path}.{key} is required")
        properties = schema.get("properties", {})
        for key, item in value.items():
            if key in properties:
                validate_value(item, properties[key], f"{path}.{key}")
            elif schema.get("additionalProperties") is False:
                raise ValueError(f"Unknown {path}.{key}")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            raise ValueError(f"{path} contains too few items")
        if len(value) > schema.get("maxItems", 10000):
            raise ValueError(f"{path} contains too many items")
        for index, item in enumerate(value):
            validate_value(item, schema.get("items", {}), f"{path}[{index}]")
    if isinstance(value, str) and not schema.get("minLength", 0) <= len(value) <= schema.get("maxLength", 1000000):
        raise ValueError(f"{path} length is out of range")
    if (isinstance(value, (int, float)) and not isinstance(value, bool)
            and (value < schema.get("minimum", float("-inf")) or value > schema.get("maximum", float("inf")))):
        raise ValueError(f"{path} is out of range")


class ToolRegistry:
    def __init__(self, tools=()):
        self.tools = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool):
        if not isinstance(tool, Tool) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]{0,127}", tool.name):
            raise ValueError("Invalid tool implementation or name")
        if tool.name in self.tools:
            raise ValueError(f"Duplicate tool name: {tool.name}")
        if not isinstance(tool.parameters, dict):
            raise TypeError("Tool parameters must be a JSON schema")
        self.tools[tool.name] = tool
        return tool

    def validate(self, name, arguments):
        tool = self.tools[name]
        inspect.signature(tool.execute).bind(**arguments)
        validate_value(arguments, tool.parameters)
        tool.validate_arguments(arguments)

    def catalog(self):
        categories = {"read_file": "filesystem", "write_file": "filesystem", "edit_file": "filesystem",
                      "grep": "localization", "glob": "localization", "bash": "execution", "agent": "orchestration",
                      "todo_write": "orchestration", "scratchpad": "memory", "memory": "memory",
                      "repository_search": "retrieval"}
        return [{"name": t.name, "description": t.description, "parameters": t.parameters,
                 "read_only": t.read_only, "concurrency_safe": t.concurrency_safe,
                 "category": categories.get(t.name, getattr(t, "category", "mcp" if t.name.startswith("mcp__") else "external")),
                 "output": "text; structured JSON where documented"} for t in self.tools.values()]

    def load_extensions(self, allowed):
        """Only operator-allowlisted installed entry points are imported."""
        if not isinstance(allowed, list) or not all(isinstance(name, str) for name in allowed):
            raise TypeError("CORECODER_TOOL_EXTENSIONS must be a JSON array of names")
        points = entry_points()
        points = points.select(group="corecoder.tools") if hasattr(points, "select") else points.get("corecoder.tools", [])
        found = {point.name: point for point in points}
        for name in allowed:
            if name not in found:
                raise ValueError(f"Tool extension not installed: {name}")
            self.register(found[name].load()())


def build_tools(config):
    from . import ALL_TOOLS
    from .bash import BashTool

    tools = [BashTool(config.sandbox, config.workspace, config.sandbox_image) if t.name == "bash" else type(t)()
             for t in ALL_TOOLS]
    registry = ToolRegistry(tools)
    registry.load_extensions(getattr(config, "tool_extensions", []))
    return list(registry.tools.values())
