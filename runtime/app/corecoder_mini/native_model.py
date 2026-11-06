"""Actual pinned LitellmModel with scoped accounting, context and registry hooks."""

import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from . import upstream as _upstream  # noqa: F401 - selects the pinned package first
from .budget import BudgetExceeded
from .context import estimate_tokens
from .native_stream import StreamReceipt, collect_response
from .provider_diagnostics import error_diagnostics, retry_decision
from .provider_parameters import completion_parameters
from .runtime import (
    Cancelled,
    check_cancelled,
    interruptible_call,
    remaining_time,
    retry_delay,
)
from .task_state import atomic_json
from .token_policy import token_components

# The dependency ships its cost/token metadata; never fetch it or send telemetry.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
import litellm
from minisweagent.models.litellm_model import LitellmModel

from .upstream import FormatError, parse_toolcall_actions

litellm.telemetry = False
litellm.suppress_debug_info = True


class ProviderFailure(RuntimeError):
    """An exhausted, bounded upstream transport attempt (no private error text)."""


@dataclass
class NativeProvider:
    model: str
    base_url: str
    api_key_env: str
    budget: object
    pricing: tuple
    settings: dict
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def estimated_cost(self):
        return self.budget.spent


class NativeModel(LitellmModel):
    """Inherit upstream query/formatting/parsing for bash, not the old SDK wrapper."""

    # One retry owner: _query handles bounded physical attempts. Disable the
    # upstream outer retry loop for every final failure, including unexpected ones.
    abort_exceptions: ClassVar[list] = [Exception, KeyboardInterrupt]

    def __init__(self, provider, directory, registry, view, events, config):
        super().__init__(model_name="openai/" + provider.model, cost_tracking="ignore_errors",
                         format_error_template=config["format_error_template"],
                         observation_template=config["observation_template"])
        self.provider, self.directory, self.registry = provider, Path(directory), registry
        self.view, self.events = view, events
        self.calls, self.input_ratio, self.request_cost = 0, 1.25, 0.0
        self.logical_calls = 0
        self.last_usage = {}

    def query(self, messages, **kwargs):
        check_cancelled()
        settings = self.provider.settings
        schemas = [t.schema() for t in self.registry.tools.values()]
        schema_tokens = estimate_tokens([{"content": json.dumps(schemas)}]) + 128
        capacity = int((settings["context_tokens"] - settings["max_output_tokens"] - 512) / self.input_ratio) - schema_tokens - 256
        self.projection = {"compacted": False}
        if self.view:
            # A context can fit the model window yet exceed the remaining task
            # reservation. Preview the same prepared payload before discarding
            # any recallable history; reserve() still owns atomic admission.
            completion = completion_parameters(settings)
            def admission(rows):
                return self.provider.budget.admission(
                    {"messages": self._prepare_messages_for_api(rows), "tools": schemas, **completion},
                    self.provider.pricing)
            wire, self.projection = self.view.project(messages, capacity, admission=admission)
        else:
            wire = [{k: v for k, v in m.items() if k != "extra"} for m in messages]
        self.logical_calls += 1
        self.logical_call_id = uuid.uuid4().hex
        self._logical_cost = 0.0
        self._logical_attempts = 0
        # Actual upstream query owns response parsing, FormatError/cost handling,
        # native assistant fields and observation formatting.
        return super().query(wire, **kwargs)

    def _query(self, messages, **kwargs):
        settings = self.provider.settings
        limit = settings.get("provider_attempts", 3)
        decision = {}
        for attempt in range(1, limit + 1):
            if attempt > 1:
                self.events.emit("provider_retry_wait", logical_call=self.logical_calls, **decision)
                try:
                    retry_delay(decision["delay_seconds"])
                except Cancelled:
                    self.events.emit("provider_retry_cancelled", logical_call=self.logical_calls)
                    raise
            response, diagnostic, decision = self._attempt(messages, attempt, **kwargs)
            if response is not None:
                if attempt > 1:
                    self.events.emit("provider_recovered", logical_call=self.logical_calls, attempts=attempt)
                return response
            if not decision["will_retry"]:
                failure = ProviderFailure(f"category={diagnostic['category']}; http_status={diagnostic.get('http_status')}; "
                                          f"attempts={attempt}; stop={decision['reason']}")
                failure.diagnostic = diagnostic
                raise failure from None

    def _attempt(self, messages, attempt, **kwargs):
        check_cancelled()
        provider, settings = self.provider, self.provider.settings
        key = os.environ.get(provider.api_key_env)
        if not key:
            raise ProviderFailure("credential_environment_missing")
        schemas = [t.schema() for t in self.registry.tools.values()]
        schema_tokens = estimate_tokens([{"content": json.dumps(schemas)}]) + 128
        estimate = estimate_tokens(messages) + schema_tokens + len(messages) * 8
        if estimate * self.input_ratio + settings["max_output_tokens"] + 256 > settings["context_tokens"]:
            raise ProviderFailure("context_capacity: no request sent")
        completion = completion_parameters(settings)
        streaming = settings.get("provider_stream", True)
        transport = {"stream": True, "stream_options": {"include_usage": True}} if streaming else {"stream": False}
        params = {"messages": messages, "tools": schemas, **completion, **transport}
        self.calls += 1
        record = self.directory / "requests" / f"{self.calls:04d}.json"
        metadata = {"estimated_input_tokens": estimate, "estimated_token_components": token_components(messages, schema_tokens),
                    "projection": self.projection, "logical_call": self.logical_calls,
                    "logical_call_id": self.logical_call_id, "attempt": attempt, "completion_parameters": completion,
                    "transport_parameters": transport,
                    "payload_sha256": hashlib.sha256(json.dumps(params, sort_keys=True, ensure_ascii=False).encode()).hexdigest()}
        atomic_json(self.directory / "private/request-views" / record.name,
                    {"messages": messages, "tools": schemas, "projection": self.projection, "model": provider.model,
                     "completion_parameters": completion, "transport_parameters": transport})
        try:
            if attempt > 1:
                provider.budget.claim_retry(settings.get("provider_retry_limit", 3))
            reservation = provider.budget.reserve(params, provider.pricing)
        except BudgetExceeded as exc:
            atomic_json(record, {**metadata, "status": "blocked_before_transport", "transport_started": False,
                                 "error_type": type(exc).__name__, "admission": exc.details,
                                 "prompt_tokens": 0, "completion_tokens": 0, "usage_complete": True})
            raise
        self._logical_attempts += 1
        atomic_json(record, {**metadata, "status": "in_flight", "transport_started": True, "started": time.time()})
        self.events.emit("request_started", request=self.calls, estimated_input_tokens=estimate)
        started, usage, response = time.monotonic(), None, None
        error, diagnostic, decision = None, None, None
        stream_receipt = StreamReceipt() if streaming else None
        stage = "provider_completion"
        try:
            timeout = min(settings["request_seconds"], remaining_time() or settings["request_seconds"])
            options = {"api_base": provider.base_url, "api_key": key, "timeout": timeout,
                       "num_retries": 0, "max_retries": 0, **kwargs}
            # Never send both limits or let caller kwargs enlarge the reserved cap.
            options.pop("max_tokens", None)
            options.pop("max_completion_tokens", None)
            options.update({**completion, **transport})
            def complete():
                if list(self.registry.tools) == ["bash"]:
                    return super(NativeModel, self)._query(messages, **options)
                # Upstream hardcodes [BASH_TOOL]; replace only that argument for extensions.
                return litellm.completion(model=self.config.model_name, messages=messages, tools=schemas, **options)
            if streaming:
                stage = "provider_stream"
                response = collect_response(complete, messages, timeout, settings["max_output_tokens"], stream_receipt)
            else:
                response = interruptible_call(complete, timeout=timeout)
            stage = "usage_extraction"
            usage = response.usage if getattr(response.usage, "prompt_tokens", 0) > 0 else None
        except BaseException as exc:  # noqa: BLE001 - persist interrupted/failed attempt before propagating
            error = exc
            diagnostic = {**error_diagnostics(exc, secrets=(key,)), "stage": stage,
                          "response_received": response is not None}
        finally:
            if usage is None and stream_receipt is not None:
                usage = stream_receipt.usage
            try:
                provider.budget.settle(reservation, usage, provider.pricing)
            except BudgetExceeded as exc:
                error = exc
                diagnostic = {**error_diagnostics(exc), "stage": "budget_settlement", "response_received": response is not None}
            receipt = reservation.receipt
            self.request_cost += receipt["cost_usd"]
            self._logical_cost += receipt["cost_usd"]
            provider.total_prompt_tokens += receipt["prompt_tokens"]
            provider.total_completion_tokens += receipt["completion_tokens"]
            self.last_usage = {"cost": receipt["cost_usd"], "prompt_tokens": receipt["prompt_tokens"],
                               "completion_tokens": receipt["completion_tokens"],
                               "cached_prompt_tokens": receipt.get("cached_prompt_tokens"),
                               "cost_basis": receipt["cost_basis"], "is_invoice": False,
                               "seconds": time.monotonic() - started,
                               "finish_reason": response.choices[0].finish_reason if response else None}
            self.input_ratio = max(self.input_ratio, receipt["prompt_tokens"] / max(1, estimate) * 1.05)
            if error:
                try:
                    seconds_remaining = remaining_time()
                except Cancelled:
                    seconds_remaining = 0
                decision = retry_decision(diagnostic, attempt, settings.get("provider_attempts", 3),
                                          provider.budget.retries_remaining(settings.get("provider_retry_limit", 3)),
                                          seconds_remaining)
                if decision["will_retry"]:
                    admission = provider.budget.admission(params, provider.pricing)
                    if not admission["allowed"]:
                        decision.update(will_retry=False, reason="shared_budget", delay_seconds=0, admission=admission)
            atomic_json(record, {**metadata, **self.last_usage, "status": "interrupted_or_failed" if error else "returned",
                                 "transport_started": True, "usage_complete": receipt["usage_complete"],
                                 **({"stream": stream_receipt.metrics()} if streaming else {}),
                                 **({"error_type": type(error).__name__, "diagnostic": diagnostic, "retry_decision": decision} if error else {})})
            self.events.emit("request_failed" if error else "request_finished", request=self.calls,
                             **({"error_type": type(error).__name__, "diagnostic": diagnostic, "retry_decision": decision} if error else self.last_usage))
        if isinstance(error, (BudgetExceeded, Cancelled, KeyboardInterrupt)):
            raise error from None
        check_cancelled()
        return (None, diagnostic, decision) if error else (response, None, None)

    def _calculate_cost(self, response):
        # Remote aliases need configured prices; no guessed LiteLLM tariff.
        return {**self.last_usage, "cost": self._logical_cost, "transport_attempts": self._logical_attempts}

    def _parse_actions(self, response):
        from .engine import SUBMIT
        from .tools.arguments import decode_arguments
        calls = response.choices[0].message.tool_calls or []
        repairs = []
        try:
            if response.choices[0].finish_reason == "length":
                raise ValueError("Output token limit reached; no tools executed. Reissue a shorter complete tool action.")
            if response.choices[0].finish_reason not in {"stop", "tool_calls", "function_call"}:
                raise ValueError("Response did not complete normally; no tools executed.")
            if list(self.registry.tools) == ["bash"]:
                actions = super()._parse_actions(response)
            else:
                if not calls:
                    raise ValueError("A registered tool call is required")
                actions = []
                for call in calls:
                    name = call.function.name
                    arguments, repair = decode_arguments(name, call.function.arguments)
                    self.registry.validate(name, arguments)
                    action = {"tool": name, "arguments": arguments, "tool_call_id": call.id}
                    if repair:
                        repairs.append((call, arguments, {**repair, "tool_call_id": call.id}))
                        action["argument_normalization"] = repair
                    if name == "bash":
                        action.update(parse_toolcall_actions([call], format_error_template=self.config.format_error_template)[0])
                    actions.append(action)
            if len(actions) > 1 and any(a.get("command", "").strip() == SUBMIT for a in actions):
                raise ValueError("Submission must be the only tool call")
            # Validate the whole batch before changing messages or allowing any
            # tools to execute. Preserve exact original arguments locally, outside
            # the model workspace; the next API history uses normalized JSON.
            if repairs:
                atomic_json(self.directory / "private/argument-normalizations" / f"{self.calls:04d}.json", {
                    "status": "validated_not_executed", "repairs": [
                        {**metadata, "original_arguments": call.function.arguments,
                         "normalized_arguments": arguments} for call, arguments, metadata in repairs]})
                for call, arguments, _ in repairs:
                    call.function.arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
                self.events.emit("tool_arguments_normalized", request=self.calls,
                                 repairs=[metadata for _, _, metadata in repairs])
            return actions
        except json.JSONDecodeError as exc:
            raise FormatError({"role": "user", "content":
                f"Invalid tool call: {call.function.name} arguments are not one JSON object "
                f"({exc.msg}, line {exc.lineno}, column {exc.colno}). "
                "Send one complete object without extra braces, another object or trailing text. "
                "No tools in this response were executed. Available tools: " + ", ".join(self.registry.tools),
                "extra": {"interrupt_type": "FormatError"}}) from exc
        except (ValueError, TypeError, KeyError) as exc:
            raise FormatError({"role": "user", "content": f"Invalid tool call: {str(exc)[:600]}. Available tools: " +
                               ", ".join(self.registry.tools), "extra": {"interrupt_type": "FormatError"}}) from exc

    def serialize(self):
        result = super().serialize()
        result["info"]["provider_usage"] = {"prompt_tokens": self.provider.total_prompt_tokens,
                                             "completion_tokens": self.provider.total_completion_tokens,
                                             "cost_usd_estimate": self.request_cost,
                                             "shared_cost_usd_estimate": self.provider.estimated_cost}
        return result
