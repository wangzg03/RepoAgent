"""Structured provider evidence without copying echoed prompts or credentials."""

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from .provider_recovery import diagnose

PARAMETERS = ("max_tokens", "max_completion_tokens", "temperature", "stream_options", "enable_thinking",
              "reasoning_content", "tool_call_id", "tool_calls", "arguments", "messages", "tools")
CODES = {"internal_error", "server_error", "service_unavailable", "invalid_request_error", "context_length_exceeded",
         "insufficient_quota", "authentication_error", "invalid_api_key", "rate_limit_exceeded", "model_not_found",
         "unsupported_parameter", "invalidparameter", "throttling", "internalerror"}


def attribute(value, name, default=None):
    try:
        return getattr(value, name, default)
    except Exception:  # noqa: BLE001 - diagnosis must not replace the original failure
        return default


def error_diagnostics(exc, secrets=()):
    chain, current, seen = [], exc, set()
    while current is not None and id(current) not in seen and len(chain) < 8:
        seen.add(id(current))
        chain.append(current)
        current = attribute(current, "__cause__") or attribute(current, "__context__")
    status, texts, codes, headers, status_chain = None, [], [], {}, []
    for error in chain:
        candidate = attribute(error, "response")
        found = attribute(error, "status_code") or attribute(candidate, "status_code")
        if status is None and type(found) is int and 100 <= found <= 599:
            status = found
        if type(found) is int and 100 <= found <= 599:
            status_chain.append({"exception_type": type(error).__name__, "http_status": found})
        candidate_headers = attribute(candidate, "headers", {}) or {}
        if hasattr(candidate_headers, "get"):
            for name in ["retry-after", "content-type", "x-request-id", "request-id", "x-correlation-id", "x-acs-request-id", "x-amzn-requestid"]:
                if candidate_headers.get(name):
                    headers[name] = candidate_headers.get(name)
        message = attribute(error, "message")
        if isinstance(message, str):
            texts.append(message[:16384])
        body = attribute(error, "body")
        if isinstance(body, (dict, list, str)):
            try:
                texts.append(json.dumps(body, ensure_ascii=False)[:16384])
            except (ValueError, TypeError):
                pass
            if isinstance(body, dict):
                record = body.get("error", body)
                if isinstance(record, dict):
                    codes += [str(record.get(k, "")).lower() for k in ("code", "type")]
        # Some wrappers only preserve the exception string. Inspect locally;
        # the text is fingerprinted, never emitted or saved.
        texts.append(str(error)[:16384])
    text = "\n".join(texts)
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[credential redacted]")
    lower = text.lower()
    category = diagnose(exc)
    if status in {401, 403}:
        category = "authentication"
    elif status == 402 or any(c == "insufficient_quota" for c in codes):
        category = "quota"
    elif status == 404:
        category = "model_or_endpoint"
    elif status == 429:
        category = "rate_limit"
    elif status in {408, 504}:
        category = "timeout"
    elif status is not None and status >= 500:
        category = "server"
    elif status in {400, 413, 422}:
        category = "request_parameters"
    elif category == "fatal" and any(type(e).__name__ in {"APIError", "JSONDecodeError", "APIResponseValidationError"} for e in chain):
        category = "provider_api" if not any(type(e).__name__ == "JSONDecodeError" for e in chain) else "invalid_response"
    # LiteLLM can wrap these local failures in a synthetic 500. A storage bound
    # is not a transient server failure and must not trigger another transport.
    if any(type(e).__name__ == "IncompleteStream" for e in chain):
        category = "invalid_response"
    elif any(type(e).__name__ == "StreamLimitExceeded" for e in chain):
        category = "response_bound"
    if category in {"request_parameters", "parameters"} and any(x in lower for x in ["context_length_exceeded", "maximum context", "context window"]):
        category = "context_capacity"
    request_id = None
    for name in ["x-request-id", "request-id", "x-correlation-id", "x-acs-request-id", "x-amzn-requestid"]:
        value = headers.get(name)
        if (isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:/-]{1,128}", value)
                and not value.lower().startswith(("sk-", "bearer")) and not any(s and s in value for s in secrets)):
            request_id = value
            break
    retry_after = None
    value = headers.get("retry-after")
    if value is not None:
        try:
            retry_after = float(value)
        except (TypeError, ValueError):
            try:
                date = parsedate_to_datetime(str(value))
                retry_after = (date - datetime.now(timezone.utc)).total_seconds()
            except (TypeError, ValueError, OverflowError):
                pass
        if retry_after is not None and (not math.isfinite(retry_after) or retry_after < 0):
            retry_after = None
    markers = []
    for marker, phrases in {
        "json_decode": ["expecting value", "jsondecodeerror", "invalid json"],
        "upstream_timeout": ["upstream timed out", "gateway timeout", "read timed out"],
        "connection_reset": ["connection reset", "remoteprotocolerror", "unexpected eof"],
        "missing_reasoning": ["reasoning_content is missing", "missing reasoning_content"],
        "unsupported_parameter": ["unsupported parameter", "unrecognized request argument"],
        "context_length": ["context_length_exceeded", "maximum context length"],
    }.items():
        if any(phrase in lower for phrase in phrases):
            markers.append(marker)
    content_type = str(headers.get("content-type", "")).split(";")[0].lower()
    if content_type not in {"application/json", "application/problem+json", "text/html", "text/plain", "text/event-stream"}:
        content_type = "other_or_unknown"
    return {"category": category, "http_status": status, "http_status_chain": status_chain,
            "exception_types": [type(e).__name__ for e in chain],
            "request_id": request_id, "content_type": content_type,
            "retry_after_seconds": retry_after, "provider_codes": sorted(set(codes) & CODES),
            "parameter_markers": [p for p in PARAMETERS if p in lower], "technical_markers": markers,
            "error_text_chars_observed": len(text), "error_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "raw_error_text_retained": False}


def retry_decision(diagnostic, attempt, limit, remaining_retries, seconds_remaining):
    eligible = diagnostic["category"] in {"rate_limit", "server", "timeout", "dns", "connection", "provider_api", "invalid_response"}
    delay = max(min(8, 2 ** attempt), diagnostic.get("retry_after_seconds") or 0)
    reason = "transient_provider_failure"
    if not eligible:
        reason = "nonrecoverable"
    elif attempt >= limit:
        reason = "attempt_limit"
    elif remaining_retries <= 0:
        reason = "task_retry_limit"
    elif delay > 30:
        reason = "retry_after_too_long"
    elif seconds_remaining is not None and seconds_remaining <= delay + 1:
        reason = "deadline_reserve"
    return {"eligible": eligible, "will_retry": reason == "transient_provider_failure", "reason": reason,
            "delay_seconds": delay if reason == "transient_provider_failure" else 0,
            "attempt": attempt, "attempt_limit": limit, "task_retries_remaining": remaining_retries}
