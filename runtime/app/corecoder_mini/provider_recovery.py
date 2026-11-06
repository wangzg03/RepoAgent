"""Bounded provider recovery; no model diagnosis, tool replay or new task budget."""

import copy
import hashlib
import json
import math
import socket
import ssl
import time
import uuid

from .budget import BudgetExceeded
from .runtime import Cancelled, check_cancelled, local, remaining_time


def diagnose(exc):
    """Return safe categories from exception types/status, never raw provider text."""
    status = getattr(exc, "status_code", None)
    if isinstance(exc, (Cancelled, KeyboardInterrupt, BudgetExceeded)):
        return "cancelled" if not isinstance(exc, BudgetExceeded) else "budget"
    if status in {401, 403}:
        return "authentication"
    if status == 404:
        return "model_or_endpoint"
    if status == 429:
        return "rate_limit"
    if status in {408, 504}:
        return "timeout"
    if isinstance(status, int) and status >= 500:
        return "server"
    if status == 400:
        return "parameters"
    chain, current = [], exc
    while current is not None and len(chain) < 8 and id(current) not in {id(e) for e in chain}:
        chain.append(current)
        current = current.__cause__ or current.__context__
    if any(isinstance(e, ssl.SSLCertVerificationError) for e in chain):
        return "tls_configuration"
    names = {type(e).__name__ for e in chain}
    if names & {"APITimeoutError", "Timeout", "TimeoutError", "ReadTimeout", "ConnectTimeout", "TimeoutException"}:
        return "timeout"
    if any(isinstance(e, socket.gaierror) for e in chain):
        return "dns"
    if names & {"APIConnectionError", "ConnectError", "RemoteProtocolError", "ConnectionError"}:
        return "connection"
    if names & {"RateLimitError"}:
        return "rate_limit"
    return "fatal"


def apply_action(params, action):
    """The entire allowlist: caps, model, endpoint, tools and messages cannot change."""
    body = params.get("extra_body") or {}
    if action == "completion_token_name" and "max_tokens" in params:
        params["max_completion_tokens"] = params.pop("max_tokens")
    elif action == "default_temperature":
        params.pop("temperature", None)
    elif action == "without_stream_usage":
        params.pop("stream_options", None)
    elif action == "without_thinking_budget":
        body.pop("thinking_budget", None)
    elif action == "without_optional_thinking":
        body["enable_thinking"] = False
        body.pop("thinking_budget", None)
    else:
        return
    if body or "extra_body" in params:
        params["extra_body"] = body


class RecoveryJournal:
    """Task-local safe receipts, shared by main/review threads through runtime context."""

    def __init__(self, harness):
        self.h = harness

    def record(self, kind, **data):
        store = self.h.store
        with store.lock:
            lease = store.directory / "lease.json"
            if not lease.exists() or json.loads(lease.read_text())["token"] != store.lease_token:
                return None  # A late transport cleanup cannot alter a released/new owner's journal.
            return store.append("provider_" + kind, **data)

    def restore_budget(self):
        from .routing import shared_budget
        budget = shared_budget(self.h.agent.llm)
        if budget is None:
            return
        entries = {}
        for event in self.h.store.events:
            if event["type"] == "provider_reserved":
                entries[event["reservation_id"]] = dict(event)
            elif event["type"] == "provider_settled" and event["reservation_id"] in entries:
                entries[event["reservation_id"]].update(event)
        if not entries and budget.requests == 0 and any(e["type"] == "model_request" for e in self.h.store.events):
            raise BudgetExceeded("Legacy task has no durable provider receipts; explicitly reconcile its prior budget before resume")
        budget.reconcile(entries.values())

    def actions(self, key):
        with self.h.store.lock:
            return list(dict.fromkeys(e["action"] for e in self.h.store.events
                if e["type"] == "provider_adjustment" and e.get("profile") == key))


class Recovery:
    def __init__(self, llm, params):
        self.limit = getattr(llm, "provider_attempts", 3)
        self.enabled = getattr(llm, "provider_recovery", True)
        if not self.enabled:
            self.limit = 1
        self.relax_thinking = getattr(llm, "recover_optional_thinking", True)
        self.qwen = llm.model.split("/")[-1].lower().startswith("qwen")
        self.journal = getattr(local, "provider_recovery", None)
        self.id = uuid.uuid4().hex
        self.start = time.monotonic()
        self.transport_failures = 0
        self.actions = []
        # Hash the endpoint/configuration, never persist credentials or raw requests.
        endpoint = str(getattr(getattr(llm, "client", None), "base_url", getattr(llm, "base_url", "")))
        config = {k: params[k] for k in ("model", "max_tokens", "max_completion_tokens", "temperature", "extra_body") if k in params}
        self.key = hashlib.sha256(json.dumps([endpoint, config, self.relax_thinking], sort_keys=True).encode()).hexdigest()
        if self.enabled and self.journal:
            for action in self.journal.actions(self.key):
                apply_action(params, action)
                self.actions.append(action)

    def record(self, kind, **data):
        if self.journal:
            return self.journal.record(kind, request_id=self.id, profile=self.key, **data)

    def begin(self, number):
        check_cancelled()
        if self.journal and not self.record("attempt", number=number, limit=self.limit):
            raise Cancelled("Task provider journal lease is no longer owned")

    def succeeded(self, number):
        self.record("recovered" if number > 1 else "succeeded", attempts=number,
                    seconds=round(time.monotonic() - self.start, 6), actions=self.actions)

    def failed(self, exc, number, params, emitted, adapt, stream_started=False):
        category = diagnose(exc)
        self.record("failure", number=number, category=category, http_status=getattr(exc, "status_code", None),
                    streamed_tool_emitted=bool(emitted))
        if emitted or number >= self.limit or not self.enabled or (stream_started and category == "parameters"):
            self.record("stopped", reason="streamed_tool_outcome" if emitted else "attempt_limit", category=category)
            return None
        action = None
        if category == "parameters":
            before = copy.deepcopy(params)
            if adapt(params, exc):
                if "max_tokens" in before and "max_tokens" not in params:
                    action = "completion_token_name"
                elif "temperature" in before and "temperature" not in params:
                    action = "default_temperature"
            else:
                # Only explicit mentions may change Qwen thinking parameters.
                message = str(exc).lower()
                body = params.get("extra_body") or {}
                if self.qwen and "thinking_budget" in message and "thinking_budget" in body:
                    action = "without_thinking_budget"
                elif (self.qwen and self.relax_thinking and "enable_thinking" in message
                      and body.get("enable_thinking") is True):
                    action = "without_optional_thinking"
                elif "stream_options" in params:
                    action = "without_stream_usage"  # Legacy OpenAI-compatible dialect fallback, once.
                else:
                    return None
        elif category in {"connection", "timeout", "dns", "server", "rate_limit"}:
            if category in {"connection", "timeout"}:
                self.transport_failures += 1
                if (self.transport_failures >= 2 and self.qwen and self.relax_thinking
                        and (params.get("extra_body") or {}).get("enable_thinking") is True):
                    action = "without_optional_thinking"
        else:
            self.record("stopped", reason="nonrecoverable", category=category)
            return None
        if action:
            # Persist the approved intent before submitting changed parameters.
            self.record("adjustment", action=action, basis="provider_rejection" if category == "parameters" else "bounded_fallback",
                        cause_proven=category == "parameters")
            apply_action(params, action)
            self.actions.append(action)
        delay = 0 if category == "parameters" else min(8, 2 ** (number - 1))
        if category == "rate_limit":
            headers = getattr(getattr(exc, "response", None), "headers", {})
            try:
                value = float(headers.get("retry-after", delay))
                if math.isfinite(value):
                    # Do not retry before the server's minimum or wait indefinitely.
                    if value > 30:
                        self.record("stopped", reason="retry_after_exceeds_window", category=category)
                        return None
                    delay = max(delay, value)
            except (ValueError, TypeError):
                pass
        remaining = remaining_time()
        if remaining is not None and remaining <= delay:
            self.record("stopped", reason="deadline_headroom", category=category)
            return None
        self.record("retry", next_attempt=number + 1, delay_seconds=delay, category=category)
        return delay


def attach(harness, restore=False):
    journal = RecoveryJournal(harness)
    if restore:
        journal.restore_budget()
    local.provider_recovery = journal


def inherit(parent, child):
    """Both kinds of bounded child retain the parent's charge/recovery journal."""
    if parent.harness and parent.harness.store and parent.harness.state:
        child._provider_journal = RecoveryJournal(parent.harness)


def model_preflight(llm, diagnostics):
    """Read-only model discovery, bounded separately; no generation/probe completion."""
    from .runtime import interruptible_call, retry_delay
    recovery = Recovery(llm, {"model": llm.model})
    for number in range(1, recovery.limit + 1):
        recovery.begin(number)
        try:
            models = interruptible_call(lambda: llm.client.models.list(
                timeout=min(llm.request_timeout, remaining_time() or llm.request_timeout)))
            model = next((item for item in models.data if item.id == llm.model), None)
            if model is None:
                raise ValueError("Configured model was not listed; do not guess another model")
            diagnostics.append({"attempt": number, "status": "available", "generation_tokens": 0})
            return {k: getattr(model, k, None) for k in ("id", "created", "owned_by")}
        except Exception as exc:
            diagnostics.append({"attempt": number, "status": "failed", "category": diagnose(exc), "generation_tokens": 0})
            delay = recovery.failed(exc, number, {}, False, lambda params, error: False)
            if delay is None:
                raise
            retry_delay(delay)
