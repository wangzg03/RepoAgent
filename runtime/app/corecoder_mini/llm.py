"""LLM provider layer - thin wrapper over OpenAI-compatible APIs.

Since most providers (DeepSeek, Qwen, Kimi, GLM, Ollama, etc.) expose an
OpenAI-compatible endpoint, we just use the openai SDK directly.  Switch
provider by changing OPENAI_BASE_URL + OPENAI_API_KEY. That's it.

For providers that are NOT OpenAI-compatible (AWS Bedrock, Google Vertex,
etc.), use the LiteLLM backend which routes to 100+ providers through a
single unified interface. Set CORECODER_PROVIDER=litellm.
"""

import copy
import json
import time  # noqa: F401 - retained for existing provider retry instrumentation
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)

from .budget import BudgetExceeded
from .runtime import (
    Cancelled,
    check_cancelled,
    interruptible_call,
    interruptible_stream,
    local,
    remaining_time,
    retry_delay,
)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str | None = None
    reasoning_content: str = ""
    cached_prompt_tokens: int | None = None

    @property
    def message(self) -> dict:
        """Convert to OpenAI message format for appending to history."""
        msg: dict = {"role": "assistant", "content": self.content or None}
        if self.tool_calls:
            # Native thinking/tool protocols (including Qwen) need this opaque
            # field back on the assistant tool-call message. Never mix it into
            # the user-facing answer or fabricate it when a provider omits it.
            if self.reasoning_content:
                msg["reasoning_content"] = self.reasoning_content
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in self.tool_calls
            ]
        return msg


# pricing per million tokens: (input, output)
# sources: openai.com/api/pricing, api-docs.deepseek.com, platform.claude.com,
#          platform.moonshot.ai, alibabacloud.com/help/en/model-studio
_PRICING = {
    # OpenAI - current flagships
    "gpt-5.5": (5, 30),
    "gpt-5.4": (2.5, 15),
    "gpt-5.4-mini": (0.75, 4.5),
    "gpt-5.4-nano": (0.2, 1.25),
    "o4-mini": (1.1, 4.4),
    # OpenAI - previous gen (still widely used)
    "gpt-4.1": (2, 8),
    "gpt-4.1-mini": (0.4, 1.6),
    "gpt-4.1-nano": (0.1, 0.4),
    "gpt-4o": (2.5, 10),
    "gpt-4o-mini": (0.15, 0.6),
    # DeepSeek
    "deepseek-chat": (0.27, 1.10),
    "deepseek-reasoner": (0.55, 2.19),
    # Anthropic Claude
    "claude-opus-4-6": (5, 25),
    "claude-sonnet-4-6": (3, 15),
    "claude-haiku-4-5": (1, 5),
    # Alibaba Qwen
    "qwen3-max": (0.78, 3.9),
    "qwen3-plus": (0.26, 0.78),
    "qwen-max": (0.78, 3.9),
    # Moonshot Kimi
    "kimi-k3": (3, 15),
    "kimi-k2.5": (0.6, 3),
}

# same shape as the table above; entries here win over the built-ins
PRICING_FILE = Path.home() / ".corecoder" / "pricing.json"


def _load_pricing() -> dict:
    """Built-in prices overlaid with the user's pricing.json, so a new model
    or a price change lands without waiting for a release. A missing or
    broken file just means the built-ins."""
    pricing = dict(_PRICING)
    try:
        overrides = json.loads(PRICING_FILE.read_text(encoding="utf-8"))
        pricing.update({m: (float(r[0]), float(r[1])) for m, r in overrides.items()})
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError):
        pass
    return pricing


def _adapt_rejected_param(params: dict, exc: BadRequestError) -> bool:
    """Translate or drop one parameter a provider rejected with a 400.

    Newer OpenAI models (gpt-5, o-series) accept only ``max_completion_tokens``
    and the default temperature; the 400 message quotes the offender
    (``'max_tokens'``). Quoted matching is load-bearing: the rejection text
    for ``max_completion_tokens`` must not re-trigger the translation.
    Returns False when nothing recognizable was rejected.
    """
    message = str(exc).lower()
    if "'max_tokens'" in message and "max_tokens" in params:
        params["max_completion_tokens"] = params.pop("max_tokens")
        return True
    if "'temperature'" in message and "temperature" in params:
        params.pop("temperature")
        return True
    return False


class LLM:
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str | None = None,
        budget=None,
        allow_text_tool_calls=False,
        pricing=None,
        request_timeout=60,
        provider_recovery=True,
        provider_attempts=3,
        recover_optional_thinking=True,
        **kwargs,
    ):
        self.model = model
        if request_timeout <= 0:
            raise ValueError("Request timeout must be positive")
        self.request_timeout = request_timeout
        self._recovery_config(provider_recovery, provider_attempts, recover_optional_thinking)
        self.client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0, timeout=request_timeout)
        self.budget = budget
        self.allow_text_tool_calls = allow_text_tool_calls
        self.pricing = pricing
        self.extra = kwargs  # temperature, max_tokens, etc.
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def _recovery_config(self, enabled, attempts, thinking):
        if type(attempts) is not int or not 1 <= attempts <= 6:
            raise ValueError("Provider attempts must be 1..6")
        self.provider_recovery = bool(enabled)
        self.provider_attempts = attempts
        self.recover_optional_thinking = bool(thinking)

    @property
    def estimated_cost(self) -> float | None:
        """Rough cost estimate in USD. Returns None if the model has no known price."""
        if getattr(self, "budget", None) is not None:
            return self.budget.spent
        pricing = getattr(self, "pricing", None)
        if pricing is None:
            pricing = _load_pricing().get(self.model)
        if not pricing:
            return None
        input_rate, output_rate = pricing
        return self.total_prompt_tokens * input_rate / 1_000_000 + self.total_completion_tokens * output_rate / 1_000_000

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        on_token=None,
        on_reasoning=None,
        on_tool_ready=None,
        tool_choice=None,
        response_format=None,
    ) -> LLMResponse:
        """Send messages, stream back response, handle tool calls.

        on_reasoning, when given, receives reasoning_content deltas from
        thinking models (deepseek-reasoner, kimi-k2-thinking, ...). Reasoning
        stays separate from answer text. Native reasoning_content is retained
        on assistant tool-call messages for providers whose tool protocol uses it.
        """
        params: dict = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            **copy.deepcopy(self.extra),
        }
        if tools:
            params["tools"] = tools
        if tool_choice is not None:
            params["tool_choice"] = tool_choice
        if response_format is not None:
            params["response_format"] = response_format

        # Provider-dialect fallbacks. A 400 that quotes a parameter is adapted
        # (newer OpenAI models take max_completion_tokens, default temperature
        # only); a 400 that names nothing drops stream_options once, matching
        # the previous single-fallback behavior for servers that reject the
        # OpenAI extension silently. The loop is bounded by the param set, and
        # an unrecognized 400 re-raises instead of doubling _call_with_retry's
        # exhausted retries. LiteLLM never lands here: drop_params strips
        # unsupported keys on that path.
        params["stream_options"] = {"include_usage": True}
        from .provider_recovery import Recovery
        recovery = Recovery(self, params)
        self._stream_executed = False
        for attempt in range(1, recovery.limit + 1):
            recovery.begin(attempt)
            stream = None
            try:
                # One aggregate cap includes create errors, parameter negotiation
                # and dropped streams. Never nest three-by-three HTTP retries.
                stream = self._call_with_retry(copy.deepcopy(params), max_retries=1)
                response = self._drain(stream, on_token, on_reasoning, on_tool_ready)
                recovery.succeeded(attempt)
                return self._text_tool_response(response, tools) if getattr(self, "allow_text_tool_calls", False) else response
            except (KeyboardInterrupt, Cancelled):
                if getattr(local, "cancel", None) is not None:
                    local.cancel.set()
                raise
            except Exception as exc:
                # No whole-task restart: completed edits/checks and the original
                # task deadline remain in place. An emitted tool forbids replay.
                delay = recovery.failed(exc, attempt, params, self._stream_executed, _adapt_rejected_param,
                                        stream_started=stream is not None)
                if delay is None:
                    raise
            finally:
                if hasattr(stream, "close"):
                    try:
                        stream.close()
                    except ValueError:
                        if not getattr(local, "cancel", None) or not local.cancel.is_set():
                            raise
            retry_delay(delay)

    @staticmethod
    def _text_tool_response(response, tools):
        """Opt-in local-provider compatibility: an entire JSON tool envelope only.

        Never extracts requests from prose, code fences or quoted documents.
        Native tool_calls take precedence. All names must be offered this turn;
        malformed envelopes stay ordinary text and are never executed.
        """
        if response.tool_calls or not response.content or not tools:
            return response
        try:
            decoder, remainder, values = json.JSONDecoder(), response.content.strip(), []
            while remainder:
                value, end = decoder.raw_decode(remainder)
                values.extend(value if isinstance(value, list) else [value])
                remainder = remainder[end:].lstrip()
        except (ValueError, TypeError):
            return response
        allowed = {tool["function"]["name"]: tool["function"] for tool in tools}
        envelopes = []
        for item in values:
            if (
                not isinstance(item, dict)
                or set(item) != {"name", "arguments"}
                or not isinstance(item["name"], str)
                or item["name"] not in allowed
            ):
                return response
            args = item["arguments"]
            if isinstance(args, list):
                # An unambiguous single-array parameter (e.g. todo_write(tasks))
                # is sometimes emitted as the array rather than {tasks: array}.
                properties = allowed[item["name"]].get("parameters", {}).get("properties", {})
                if len(properties) != 1 or next(iter(properties.values())).get("type") != "array":
                    return response
                args = {next(iter(properties)): args}
            if not isinstance(args, dict):
                return response
            envelopes.append({"name": item["name"], "arguments": args})
        if not envelopes:
            return response
        adapted = LLMResponse(
            content="",
            tool_calls=[ToolCall("json_" + uuid.uuid4().hex, item["name"], item["arguments"]) for item in envelopes],
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            finish_reason=response.finish_reason,
            reasoning_content=response.reasoning_content,
            cached_prompt_tokens=response.cached_prompt_tokens,
        )
        adapted.provider_tool_text = response.content
        return adapted

    def _drain(self, stream, on_token, on_reasoning, on_tool_ready=None) -> LLMResponse:
        """Consume one stream into an LLMResponse."""
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tc_map: dict[int, dict] = {}  # index -> {id, name, arguments_str}
        prompt_tok = 0
        completion_tok = 0
        cached_prompt_tok = None
        finish_reason = None
        emitted = set()
        self._stream_executed = False

        def emit(idx):
            if not on_tool_ready or idx in emitted:
                return
            raw = tc_map[idx]
            try:
                args = json.loads(raw["args"])
            except (ValueError, KeyError):
                return
            if raw["id"] and raw["name"] and isinstance(args, dict):
                emitted.add(idx)
                self._stream_executed = True
                on_tool_ready(ToolCall(raw["id"], raw["name"], args))

        for chunk in interruptible_stream(stream):
            check_cancelled()
            # usage info comes in the final chunk; getattr-style reads stay safe
            # across OpenAI SDK objects and litellm's provider-varying shapes
            usage = getattr(chunk, "usage", None)
            if usage:
                # some providers send usage with null fields; coerce to 0 so the
                # running totals below don't blow up on int + None
                prompt_tok = getattr(usage, "prompt_tokens", 0) or 0
                completion_tok = getattr(usage, "completion_tokens", 0) or 0
                details = getattr(usage, "prompt_tokens_details", None)
                cached = details.get("cached_tokens") if isinstance(details, dict) else getattr(details, "cached_tokens", None)
                cached_prompt_tok = cached if type(cached) is int and 0 <= cached <= prompt_tok else None

            if not getattr(chunk, "choices", None):
                continue
            delta = chunk.choices[0].delta
            finish_reason = getattr(chunk.choices[0], "finish_reason", None) or finish_reason

            # Keep native tool-protocol state even without a display callback.
            # It remains separate from ordinary answer content.
            # reasoning_content is the DeepSeek/Kimi dialect, reasoning is
            # what aggregators like OpenRouter normalize it to
            native_reasoning = getattr(delta, "reasoning_content", None)
            if isinstance(native_reasoning, str):
                reasoning_parts.append(native_reasoning)
            reasoning = native_reasoning or getattr(delta, "reasoning", None)
            if reasoning and on_reasoning:
                on_reasoning(reasoning)

            # accumulate text
            if getattr(delta, "content", None):
                content_parts.append(delta.content)
                if on_token:
                    on_token(delta.content)

            # accumulate tool calls across chunks
            if getattr(delta, "tool_calls", None):
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    # A later call index seals the preceding call. Never run
                    # merely because an incomplete JSON prefix happens to parse.
                    for previous in sorted(tc_map):
                        if previous < idx:
                            emit(previous)
                    if idx not in tc_map:
                        tc_map[idx] = {"id": "", "name": "", "args": ""}
                    if tc_delta.id:
                        tc_map[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tc_map[idx]["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            tc_map[idx]["args"] += tc_delta.function.arguments
            if getattr(chunk.choices[0], "finish_reason", None) == "tool_calls":
                for idx in sorted(tc_map):
                    emit(idx)

        # parse accumulated tool calls
        parsed: list[ToolCall] = []
        for idx in sorted(tc_map):
            raw = tc_map[idx]
            try:
                args = json.loads(raw["args"])
            except (json.JSONDecodeError, KeyError):
                raise ValueError(f"Invalid JSON arguments for tool {raw['name']}") from None
            if not isinstance(args, dict):
                raise TypeError("Tool arguments must be a JSON object")
            parsed.append(ToolCall(id=raw["id"], name=raw["name"], arguments=args))

        self.total_prompt_tokens += prompt_tok
        self.total_completion_tokens += completion_tok

        return LLMResponse(
            content="".join(content_parts),
            tool_calls=parsed,
            prompt_tokens=prompt_tok,
            completion_tokens=completion_tok,
            finish_reason=finish_reason,
            reasoning_content="".join(reasoning_parts),
            cached_prompt_tokens=cached_prompt_tok,
        )

    def _call_with_retry(self, params: dict, max_retries: int = 3):
        """Retry on transient errors with exponential backoff."""
        for attempt in range(max_retries):
            try:
                check_cancelled()
                remaining = remaining_time()
                if remaining is not None:
                    params["timeout"] = min(getattr(self, "request_timeout", 60), remaining)
                return self._budgeted_create(params, lambda: interruptible_call(lambda: self.client.chat.completions.create(**params)))
            except (RateLimitError, APITimeoutError, APIConnectionError):
                if attempt == max_retries - 1:
                    raise
                retry_delay(2**attempt)
            except APIError as e:
                status_code = getattr(e, "status_code", None)
                if status_code and status_code >= 500 and attempt < max_retries - 1:
                    retry_delay(2**attempt)
                else:
                    raise

    def _budgeted_create(self, params, create):
        budget = getattr(self, "budget", None)
        if budget is None:
            return create()
        rates = getattr(self, "pricing", None)
        if rates is None:
            rates = _load_pricing().get(params["model"])
        amount = budget.reserve(params, rates)
        journal = getattr(local, "provider_recovery", None)
        if journal:
            try:
                if not journal.record("reserved", reservation_id=amount.identity,
                                      reserved_usd=float(amount), reserved_tokens=amount.tokens):
                    raise Cancelled("Provider submission requires the active task journal lease")
            except BaseException:
                budget.settle(amount, type("Usage", (), {"prompt_tokens": 0, "completion_tokens": 0})(), rates)
                raise  # Do not submit an unjournaled request.

        def settle(usage):
            try:
                budget.settle(amount, usage, rates)
            finally:
                if journal:
                    journal.record("settled", reservation_id=amount.identity, receipt=amount.receipt)

        try:
            stream = create()
        except BadRequestError:
            settle(type("Usage", (), {"prompt_tokens": 0, "completion_tokens": 0})())
            raise
        except BaseException:
            settle(None)
            raise

        def metered():
            usage = None
            try:
                for chunk in stream:
                    usage = getattr(chunk, "usage", None) or usage
                    yield chunk
            finally:
                try:
                    if hasattr(stream, "close"):
                        stream.close()
                finally:
                    settle(usage)

        return metered()


class LiteLLM(LLM):
    """LLM backend via LiteLLM, supporting 100+ providers.

    Use this when your target provider is NOT OpenAI-compatible
    (AWS Bedrock, Google Vertex, Cohere, etc.) or when you want
    a single interface to switch between any provider by changing
    the model string.

    Set CORECODER_PROVIDER=litellm and use LiteLLM model strings
    like ``anthropic/claude-3-haiku``, ``bedrock/anthropic.claude-v2``,
    ``vertex_ai/gemini-pro``, etc.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        budget=None,
        allow_text_tool_calls=False,
        pricing=None,
        request_timeout=60,
        provider_recovery=True,
        provider_attempts=3,
        recover_optional_thinking=True,
        **kwargs,
    ):
        # skip LLM.__init__ which creates an OpenAI client
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.request_timeout = request_timeout
        self._recovery_config(provider_recovery, provider_attempts, recover_optional_thinking)
        self.extra = kwargs
        self.budget = budget
        self.allow_text_tool_calls = allow_text_tool_calls
        self.pricing = pricing
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def _call_with_retry(self, params: dict, max_retries: int = 3):
        """Retry on transient errors with exponential backoff via litellm."""
        import litellm

        params["drop_params"] = True
        if self.api_key:
            params["api_key"] = self.api_key
        if self.base_url:
            params["api_base"] = self.base_url

        for attempt in range(max_retries):
            try:
                check_cancelled()
                params["timeout"] = min(self.request_timeout, remaining_time() or self.request_timeout)
                return self._budgeted_create(params, lambda: interruptible_call(lambda: litellm.completion(**params)))
            except Cancelled:
                raise
            except Exception as e:
                err = str(e).lower()
                is_transient = any(kw in err for kw in ["rate_limit", "timeout", "connection", "502", "503", "529"])
                is_server = any(kw in err for kw in ["500", "502", "503", "504"])
                if (is_transient or is_server) and attempt < max_retries - 1:
                    retry_delay(2**attempt)
                else:
                    raise


class ModelRouter:
    """Ordered provider fallback, with a single budget shared by all routes."""

    def __init__(self, routes):
        if not routes:
            raise ValueError("At least one model route is required")
        self.routes = routes
        from .budget import Budget

        budgets = [getattr(route, "budget", None) for route in routes]
        shared = next((budget for budget in budgets if isinstance(budget, Budget)), None)
        if shared is not None and any(budget is not shared for budget in budgets):
            raise ValueError("All fallback routes must share the same budget")

    @property
    def model(self):
        return self.routes[0].model

    @model.setter
    def model(self, value):
        self.routes[0].model = value

    @property
    def total_prompt_tokens(self):
        budget = getattr(self.routes[0], "budget", None)
        if budget is not None:
            return budget.prompt_tokens
        return sum(r.total_prompt_tokens for r in self.routes)

    @property
    def total_completion_tokens(self):
        budget = getattr(self.routes[0], "budget", None)
        if budget is not None:
            return budget.completion_tokens
        return sum(r.total_completion_tokens for r in self.routes)

    @property
    def estimated_cost(self):
        budget = getattr(self.routes[0], "budget", None)
        if budget is not None:
            return budget.spent
        costs = [r.estimated_cost for r in self.routes]
        return None if any(c is None for c in costs) else sum(costs)

    def fork(self):
        import copy

        return ModelRouter([copy.copy(r) for r in self.routes])

    def chat(self, **kwargs):
        emitted = False
        callback = kwargs.get("on_tool_ready")

        def ready(tc):
            nonlocal emitted
            emitted = True
            callback(tc)

        if callback:
            kwargs["on_tool_ready"] = ready
        for index, route in enumerate(self.routes):
            try:
                return route.chat(**kwargs)
            except BudgetExceeded:
                raise
            except Exception as exc:
                from .runtime import Cancelled

                transient = isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError))
                status = getattr(exc, "status_code", 0) or 0
                transient |= status in {408, 429} or status >= 500
                if isinstance(exc, Cancelled) or emitted or not transient or index == len(self.routes) - 1:
                    raise
