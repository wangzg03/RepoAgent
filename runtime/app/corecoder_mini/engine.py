"""One upstream loop. Adapters provide transport, Docker, and durable I/O only."""

import json
import subprocess
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import yaml

from .runtime import check_cancelled
from .sandbox import DockerSandbox
from .task_state import atomic_json
from .tools.base import Tool
from .tools.registry import ToolRegistry
from .upstream import (
    BASH_TOOL,
    DefaultAgent,
    FormatError,
    Submitted,
    format_toolcall_observation_messages,
    parse_toolcall_actions,
)

SUBMIT = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "vendor/minisweagent/config/mini.yaml"


def templates():
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


class DurableAgent(DefaultAgent):
    """Only persistence differs. Execution methods are inherited, unchanged."""

    def save(self, path, *extra_dicts):
        data = self.serialize(*extra_dicts)
        if path:
            atomic_json(Path(path), data)
        if self.messages and self.messages[-1].get("role") == "exit" and not getattr(self, "_terminal_event_sent", False):
            self._terminal_event_sent = True
            if getattr(self, "events", None):
                self.events.emit("session_finished", exit_status=self.messages[-1].get("extra", {}).get("exit_status"))
            if hasattr(self.env, "cleanup"):
                self.env.cleanup()
        return data


class ProviderModel:
    def __init__(self, llm, directory, *, context_tokens=16384, max_output_tokens=1024,
                 registry=None, view=None, events=None):
        self.llm = llm
        self.directory = Path(directory)
        self.context_tokens = context_tokens
        self.max_output_tokens = max_output_tokens
        self.config = templates()["model"]
        self.calls = 0
        self.request_cost = 0.0
        self.registry, self.view, self.events = registry, view, events
        if registry and list(registry.tools) != ["bash"]:
            from .prompts import EXTENDED_FORMAT_ERROR
            self.config["format_error_template"] = EXTENDED_FORMAT_ERROR
        self.input_ratio = 1.25

    @staticmethod
    def format_message(**kwargs):
        return kwargs

    def query(self, messages):
        from .context import estimate_tokens
        from .token_policy import token_components

        check_cancelled()
        schemas = [t.schema() for t in self.registry.tools.values()] if self.registry else [BASH_TOOL]
        schema_tokens = estimate_tokens([{"content": json.dumps(schemas)}]) + 128
        projection = {"compacted": False}
        if self.view:
            capacity = int((self.context_tokens - self.max_output_tokens - 512) / self.input_ratio) - schema_tokens - 256
            wire, projection = self.view.project(messages, capacity)
        else:
            wire = [{k: v for k, v in m.items() if k != "extra"} for m in messages]
        estimate = estimate_tokens(wire) + schema_tokens + len(wire) * 8
        attribution = token_components(wire, schema_tokens)
        if estimate * self.input_ratio + self.max_output_tokens + 256 > self.context_tokens:
            raise RuntimeError("context_capacity: full history exceeds the frozen context; no request sent")
        self.calls += 1
        intent = self.directory / "requests" / f"{self.calls:04d}.json"
        atomic_json(self.directory / "private/request-views" / f"{self.calls:04d}.json",
                    {"messages": wire, "tools": schemas, "projection": projection, "model": self.llm.model})
        atomic_json(intent, {"status": "in_flight", "estimated_input_tokens": estimate,
                             "estimated_token_components": attribution, "projection": projection, "started": time.time()})
        if self.events:
            self.events.emit("request_started", request=self.calls, estimated_input_tokens=estimate)
        started = time.monotonic()
        previous = self.llm.estimated_cost or 0
        try:
            response = self.llm.chat(wire, tools=schemas)
        except BaseException as exc:
            atomic_json(intent, {"status": "interrupted_or_failed", "error_type": type(exc).__name__,
                                 "seconds": time.monotonic() - started,
                                 "estimated_input_tokens": estimate, "estimated_token_components": attribution,
                                 "projection": projection,
                                 "usage_complete": False})
            if self.events:
                self.events.emit("request_failed", request=self.calls, error_type=type(exc).__name__)
            raise
        rates = getattr(self.llm, "pricing", None)
        usage_complete = response.prompt_tokens > 0
        if rates is not None:
            from .accounting import usage_charge
            from .budget import Budget
            charge = usage_charge({"prompt_tokens": response.prompt_tokens, "completion_tokens": response.completion_tokens,
                                   "prompt_cache_hit_tokens": getattr(response, "cached_prompt_tokens", None)}, rates) if usage_complete else None
            cost = charge["cost_usd"] if charge else float(Budget.quote({"messages": wire, "tools": schemas,
                                                                       "max_tokens": self.max_output_tokens}, rates))
        else:
            cost = (self.llm.estimated_cost or 0) - previous
        self.request_cost += cost
        extra = {"cost": cost,
                 "prompt_tokens": response.prompt_tokens, "completion_tokens": response.completion_tokens,
                 "cached_prompt_tokens": getattr(response, "cached_prompt_tokens", None),
                 "seconds": time.monotonic() - started, "finish_reason": response.finish_reason}
        self.input_ratio = max(self.input_ratio, response.prompt_tokens / max(1, estimate) * 1.05)
        atomic_json(intent, {"status": "returned", **extra, "estimated_input_tokens": estimate,
                             "estimated_token_components": attribution,
                             "projection": projection, "usage_complete": usage_complete})
        if self.events:
            self.events.emit("request_finished", request=self.calls, **extra)
        calls = [SimpleNamespace(id=tc.id, function=SimpleNamespace(
            name=tc.name, arguments=json.dumps(tc.arguments))) for tc in response.tool_calls]
        try:
            if not self.registry or list(self.registry.tools) == ["bash"]:
                actions = parse_toolcall_actions(calls, format_error_template=self.config["format_error_template"],
                                                template_kwargs={"finish_reason": response.finish_reason})
            else:
                if not calls:
                    raise ValueError("A registered tool call is required; use bash for source changes/tests and the standalone submission command to finish.")
                actions = []
                for call in calls:
                    name, arguments = call.function.name, json.loads(call.function.arguments)
                    self.registry.validate(name, arguments)
                    action = {"tool": name, "arguments": arguments, "tool_call_id": call.id}
                    if name == "bash":
                        # Keep the upstream parser for native bash calls.
                        action.update(parse_toolcall_actions([call], format_error_template=self.config["format_error_template"])[0])
                    actions.append(action)
            if len(actions) > 1 and any(a.get("command", "").strip() == SUBMIT for a in actions):
                raise ValueError("Submission must be the only tool call in the response; complete other actions first.")
        except (ValueError, KeyError, TypeError) as exc:
            available = list(self.registry.tools) if self.registry else ["bash"]
            raise FormatError({"role": "user", "content": f"Invalid tool call ({type(exc).__name__}): {str(exc)[:600]}. Available tools: " +
                               ", ".join(available) + ". Use their declared schemas; bash remains available for edits and tests.",
                               "extra": {**extra, "interrupt_type": "FormatError"}}) from exc
        except FormatError as exc:
            # Upstream charges parse failures through this same exception message.
            exc.messages[0].setdefault("extra", {}).update(extra)
            raise
        return {**response.message, "extra": {**extra, "actions": actions}}

    def format_observation_messages(self, message, outputs, template_vars=None):
        return format_toolcall_observation_messages(
            actions=message["extra"]["actions"], outputs=outputs,
            observation_template=self.config["observation_template"], template_vars=template_vars)

    def get_template_vars(self, **kwargs):
        return {"model": self.llm.model, **kwargs}

    def serialize(self):
        return {"info": {"provider_usage": {"prompt_tokens": self.llm.total_prompt_tokens,
                                            "completion_tokens": self.llm.total_completion_tokens,
                                            "cost_usd_estimate": self.request_cost,
                                            "shared_cost_usd_estimate": self.llm.estimated_cost}}}


class ScopedDocker(DockerSandbox):
    def __init__(self, root, image, job_id, read_only=False):
        super().__init__(root, image, read_only=read_only)
        self.job_id = job_id

    def command(self, command, cwd, name, timeout):
        argv = super().command(command, cwd, name, timeout)
        argv[2:2] = ["--label", "corecoder.eval.job=" + self.job_id]
        replacements = {"--memory=512m": "--memory=2g", "--memory-swap=512m": "--memory-swap=2g",
                        "--cpus=1": "--cpus=2", "/tmp:rw,nosuid,nodev,size=64m,mode=1777":
                        "/tmp:rw,exec,nosuid,nodev,size=512m,mode=1777"}
        return [replacements.get(x, x) for x in argv]


class BashTool(Tool):
    name = "bash"
    description = BASH_TOOL["function"]["description"]
    parameters = BASH_TOOL["function"]["parameters"]

    def __init__(self, sandbox, timeout, prefix="", allow_submission=True):
        self.sandbox, self.timeout, self.prefix = sandbox, timeout, prefix
        self.allow_submission = allow_submission

    def execute(self, command):
        try:
            options = {"submission_requested": self.allow_submission and command.strip() == SUBMIT} if getattr(self.sandbox, "native_submission", False) else {}
            proc = self.sandbox.run(self.prefix + command, self.sandbox.root, self.timeout, **options)
            return {"returncode": proc.returncode, "output": proc.stdout + proc.stderr, "exception_info": ""}
        except subprocess.TimeoutExpired:
            return {"returncode": 124, "output": "Command exceeded its execution deadline.",
                    "exception_info": "tool_timeout"}


class ShellEnvironment:
    def __init__(self, sandbox, directory, *, timeout=180, tool_limit=48, prefix="", events=None, depth=0):
        self.directory = Path(directory)
        self.registry = ToolRegistry([BashTool(sandbox, timeout, prefix, allow_submission=not depth)])
        self.actions = 0
        self.tool_limit = tool_limit
        self.events, self.depth = events, depth
        self.progress = None
        self.public_check_command = None

    def execute(self, action, cwd=""):
        check_cancelled()
        if self.actions >= self.tool_limit:
            raise RuntimeError("tool_budget_exhausted")
        name = action.get("tool", "bash")
        arguments = action.get("arguments", {"command": action.get("command", "")})
        self.registry.validate(name, arguments)
        if (getattr(self, 'task_contract_path', None) and not self.task_contract_path.exists()
                and name not in {'task_contract', 'read_file', 'code_search', 'trajectory_read'}):
            raise ValueError('Call task_contract once with the complete intent JSON before execution; no action ran.')
        if getattr(self, 'repoagent_settings', None):
            from repoagent.permissions import authorize
            authorize(self, name, arguments)
        self.actions += 1
        action_id = self.actions
        record = self.directory / "actions" / f"{action_id:04d}.json"
        atomic_json(record, {"status": "intent", "tool": name, "action": action, "time": time.time()})
        if self.events:
            self.events.emit("tool_started", action=self.actions, tool=name)
        standalone_submit = name == "bash" and arguments["command"].strip() == SUBMIT and not self.depth
        if standalone_submit and self.public_check_command:
            # A public check is an ordinary journaled sandbox tool, charged to the
            # same tool cap. Never run the independent hidden official evaluator here.
            checked = self.execute({"tool": "bash", "arguments": {"command": self.public_check_command},
                                    "tool_call_id": f"public-submit-check-{action_id}"})
            if checked["returncode"] != 0:
                result = {**checked, "output": "Submission deferred: configured PUBLIC check failed. Repair this failure and resubmit.\n" + checked["output"]}
                atomic_json(record, {"status": "submission_rejected", "tool": name, "action": action,
                                     "result": result, "public_check_action": self.actions})
                if self.events:
                    self.events.emit("tool_finished", action=action_id, tool=name, returncode=result["returncode"], submitted=False)
                return result
        native_submit = getattr(getattr(self.registry.tools[name], "sandbox", None), "native_submission", False)
        if standalone_submit and not native_submit:
            atomic_json(self.directory / "submission.json", {"model_submitted": True,
                        "tool_call_id": action.get("tool_call_id"), "time": time.time()})
            atomic_json(record, {"status": "submitted", "tool": name, "action": action})
            if self.events:
                self.events.emit("tool_finished", action=action_id, tool=name, returncode=0, submitted=True)
            raise Submitted({"role": "exit", "content": "Model submitted its workspace.",
                             "extra": {"exit_status": "Submitted", "submission": "workspace"}})
        started = time.monotonic()
        try:
            reused = self.progress.reused_read(name, arguments) if self.progress else None
            scoped_check = (getattr(self, 'repoagent_settings', {}).get('write_scope', ['.']) != ['.']
                            and name == 'bash' and arguments['command'] == self.public_check_command)
            tool = self.registry.tools['run_tests'] if scoped_check else self.registry.tools[name]
            raw = reused if reused is not None else tool.execute(**arguments)
            result = {"exception_info": "", "output": "", **raw} if isinstance(raw, dict) and "returncode" in raw else {
                "returncode": 0, "output": str(raw), "exception_info": ""}
            if type(result["returncode"]) is not int or not isinstance(result["output"], str):
                raise TypeError("tool_result_requires_integer_returncode_and_text_output")
        except Submitted:
            if standalone_submit:
                atomic_json(self.directory / "submission.json", {"model_submitted": True,
                            "tool_call_id": action.get("tool_call_id"), "time": time.time()})
            atomic_json(record, {"status": "submitted", "tool": name, "action": action})
            if self.events:
                self.events.emit("tool_finished", action=action_id, tool=name, returncode=0, submitted=True)
            raise
        except (ValueError, KeyError, TypeError) as exc:
            result = {"returncode": 2, "output": str(exc)[:1200], "exception_info": type(exc).__name__}
        except BaseException as exc:
            # Never replay a pending or interrupted command on restart.
            atomic_json(record, {"status": "interrupted_or_failed", "tool": name, "action": action,
                                 "error_type": type(exc).__name__, "may_have_side_effects": True})
            if self.events:
                self.events.emit("tool_failed", action=action_id, tool=name, error_type=type(exc).__name__)
            raise
        annotation = {}
        if self.progress:
            annotation = self.progress.observe(action_id, name, arguments, result)
        atomic_json(record, {"status": "reused" if result.get("reused_from") else "executed",
                             "tool": name, "action": action, "result": result, **annotation,
                             "seconds": time.monotonic() - started})
        if self.events:
            self.events.emit("tool_finished", action=action_id, tool=name, returncode=result["returncode"],
                             seconds=time.monotonic() - started)
        # Keep full local receipt. Bound only the provider observation.
        if len(result["output"]) > 18000:
            result = {**result, "output": result["output"][:9000] + "\n... bounded provider observation ...\n" + result["output"][-9000:]}
        if "trajectory_read" in self.registry.tools:
            result = {**result, "extra": {**result.get("extra", {}), **annotation, "action_id": action_id}}
            from .reference_runtime import POLICY, compact_observation
            output_limit = POLICY["tool_output"]["max_chars"]
            structured_file_result = name in {"read_file", "edit_file", "write_file"}
            if not structured_file_result and result["returncode"] == 0 and len(result["output"]) > output_limit:
                result["output"] = compact_observation(result["output"], output_limit)
            if not structured_file_result and len(result["output"]) > 4000:
                result["output"] = f"[Captured output: trajectory_read action=output action_id={action_id} offset=0 limit=4000]\n" + result["output"]
        return result

    def get_template_vars(self, **kwargs):
        return {"system": "Linux", "release": "Docker sandbox", "version": "isolated",
                "machine": "x86_64", **kwargs}

    def serialize(self):
        sandbox = self.registry.tools["bash"].sandbox
        return {"info": {"environment": {"actions": self.actions, "host_execution": False,
                                         "backend": "pinned_DockerEnvironment" if getattr(sandbox, "native_submission", False) else "injected"}}}

    def cleanup(self):
        for client in getattr(self, 'mcp_clients', []):
            client.close()
        sandbox = self.registry.tools["bash"].sandbox
        if hasattr(sandbox, "cleanup"):
            sandbox.cleanup()


def build_agent(llm, workspace, directory, image, settings, job_id=None, *, depth=0, child_provider_factory=None):
    from .children import ChildManager, ChildReportTool, SubagentTool
    from .context_view import ContextView, TrajectoryTool
    from .continuation import ActionContinuation
    from .events import Events
    from .native_docker import SessionDocker
    from .progress import WorkspaceProgress
    from .prompts import EXTENDED_SYSTEM, EXTENDED_TASK, TOOL_GUIDANCE
    from .state_tools import ProjectMemoryTool, ScratchpadTool, TodoTool, WorkingState
    from .token_policy import RuntimeGuidance
    from .tools.workspace import EditFileTool, ReadFileTool, WriteFileTool

    directory = Path(directory).resolve()
    if directory.is_relative_to(Path(workspace).resolve()):
        raise ValueError("agent_state_must_be_outside_sandbox_workspace")
    directory.mkdir(parents=True, exist_ok=True)
    events = Events(directory)
    state = WorkingState(directory)
    if not depth and settings.get('task_intent'):
        state.bind_task_contract(settings['task_intent'], settings.get('task_intent_source', 'operator'))
    identity = job_id or uuid.uuid4().hex
    settings = {**settings, "root_job_id": settings.get("root_job_id", settings.get("job_id", identity)), "job_id": identity}
    environment = ShellEnvironment(SessionDocker(workspace, image, settings["job_id"], read_only=bool(depth) and not settings.get("child_can_write", False),
                                                 owner_job_id=settings["root_job_id"]), directory,
                                   timeout=settings["tool_seconds"], tool_limit=settings["tool_limit"],
                                   prefix=settings.get("shell_prefix", ""), events=events, depth=depth)
    environment.public_check_command = settings.get("public_check_command") if not depth else None
    enabled = settings.get("harness_tools", ["scratchpad", "todo_write", "memory", "trajectory_read", "subagent"])
    extended_requested = bool(enabled or settings.get("tool_extensions"))
    if extended_requested and not depth:
        environment.progress = WorkspaceProgress(workspace, directory, settings, state)
    if extended_requested:
        sandbox = environment.registry.tools["bash"].sandbox
        file_tools = [ReadFileTool] if depth and not settings.get("child_can_write", False) else [ReadFileTool, EditFileTool, WriteFileTool]
        for tool_class in file_tools:
            environment.registry.register(tool_class(sandbox, settings["tool_seconds"], settings.get("shell_prefix", "")))
    candidates = {"scratchpad": ScratchpadTool(state), "todo_write": TodoTool(state, environment.progress),
                  "trajectory_read": TrajectoryTool(directory)}
    if "memory" in enabled:
        candidates["memory"] = ProjectMemoryTool(settings.get("memory_path", str(directory / "project-memory.sqlite3")),
                                                 settings.get("project_scope", workspace),
                                                 enabled=not depth and settings.get("task_kind", "user") != "benchmark",
                                                 settings=settings)
    children = None
    if not depth and "subagent" in enabled and settings.get("max_subagents", 2):
        children = ChildManager(llm, workspace, directory, image, settings, state, events, provider_factory=child_provider_factory)
        candidates["subagent"] = SubagentTool(children)
    for name in enabled:
        if name in {"read_file", "edit_file", "write_file"} and name in environment.registry.tools:
            continue
        if name in candidates:
            environment.registry.register(candidates[name])
        elif name != "subagent":
            raise ValueError("unknown_harness_tool")
    if depth:
        environment.registry.register(ChildReportTool(directory))
    else:
        environment.registry.load_extensions(settings.get("tool_extensions", []))
    if settings.get('repoagent'):
        from repoagent.integration import extend
        extend(state, environment, settings, workspace, directory, depth)
    extended = list(environment.registry.tools) != ["bash"]
    view = ContextView(directory, state, events, target_tokens=settings.get("history_target_tokens", 8000),
                       keep_recent=settings.get("keep_recent_turns", 4),
                       history_output_chars=settings.get("history_output_chars", 2000),
                       allow_recall="trajectory_read" in environment.registry.tools,
                       progress=environment.progress.feedback if environment.progress else None,
                       recovery=environment.progress.request_feedback if environment.progress else None,
                       guidance=RuntimeGuidance(environment, llm, settings) if settings.get("budget_guidance", True) else None) if extended else None
    if view and not depth and "memory" in candidates and settings.get("memory_startup_recall", True):
        view.project_memory = candidates["memory"].context
    if view and environment.progress:
        view.continuation = ActionContinuation(environment.progress, environment.registry, budget_provider=view.guidance)
    config = templates()["agent"]
    instance = config["instance_template"]
    if depth:
        instance = "Read-only investigation assignment (task data):\n{{task}}\nUse read_file or a focused, preferably batched bash investigation in /workspace, then subagent_report with actual action numbers. Reserve a request for that report. The complete original user request is available via trajectory_read action=request. Do not change source, delegate, or issue the parent submission command."
    elif extended:
        instance = EXTENDED_TASK + "\n" + "\n".join(
            text for name, text in TOOL_GUIDANCE.items() if name in environment.registry.tools)
        if state.data.get('task_contract'):
            from .prompts import task_directive
            instance = task_directive(state.requested_kind()) + '\n\n' + instance
    if environment.public_check_command:
        instance += "\nA configured public check runs on submission in the same sandbox; failures return here for repair. This is separate from hidden official scoring.\n"
    if settings.get('repoagent'):
        if depth:
            instance = "Isolated nonrecursive child assignment (JSON task data):\n{{task}}\nUse only granted capabilities. Write only within granted write_scope in your snapshot. Do not submit the parent task. Finish with subagent_report referencing actual tool actions."
        instance = instance.replace('Then run a relevant real\ntest/assertion with bash',
                                    'Then run a relevant real\ntest/assertion with run_tests')
        instance += """
RepoAgent: first call task_contract to record the full intent, metrics, checks and plan.
Do not merely describe a JSON plan: call the tool. Inferred metrics are proposals,
never claimed measurements. After recording the contract, implement and verify.
Use code_search for unknown definitions/references/paths; read relevant windows.
Avoid redundant searches when the task already supplies the exact target file.
Use run_tests for executable checks. For scoped tasks, arbitrary bash is disabled;
file writes use edit_file/write_file and tests run in a disposable workspace copy.
Child write permissions are limited by operator policy and isolated snapshots;
review and explicitly integrate a child patch, then test main's current workspace.
""" if not depth else "\nUse code_search for focused source localization when useful."
    from .native_model import NativeModel, NativeProvider
    if isinstance(llm, NativeProvider):
        llm.settings = settings
        native_config = templates()["model"]
        if extended:
            from .prompts import EXTENDED_FORMAT_ERROR
            native_config["format_error_template"] = EXTENDED_FORMAT_ERROR
        model = NativeModel(llm, directory, environment.registry, view, events, native_config)
    else:
        # Controlled providers remain a dependency-injection seam for tests.
        model = ProviderModel(llm, directory, context_tokens=settings["context_tokens"],
                              max_output_tokens=settings["max_output_tokens"], registry=environment.registry,
                              view=view, events=events)
    agent = DurableAgent(model, environment,
                        system_template=EXTENDED_SYSTEM if extended and not depth else config["system_template"], instance_template=instance,
                        step_limit=settings["step_limit"], cost_limit=settings["dollar_limit"],
                        wall_time_limit_seconds=settings["agent_seconds"], output_path=directory / "trajectory.json")
    if not depth and state.data.get('task_contract'):
        agent.extra_template_vars['declared_task_intent'] = state.requested_kind()
    agent.working_state, agent.children, agent.events = state, children, events
    if view and not depth:
        view.goal_provider = lambda: agent.extra_template_vars.get("task", "")
    if children:
        children.goal_provider = lambda: agent.extra_template_vars.get("task", "")
    events.emit("session_started", depth=depth, tools=list(environment.registry.tools))
    return agent
