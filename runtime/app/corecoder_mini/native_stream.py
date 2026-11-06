"""Collect a complete native response before upstream parsing or any tool effect.

Uses LiteLLM's existing stream builder, not a second SSE/tool JSON protocol.
The final usage event is authoritative; builder estimates are never billed.
"""

import json
import time
from dataclasses import dataclass

from .accounting import field
from .runtime import check_cancelled, interruptible_call, interruptible_stream


class IncompleteStream(RuntimeError):
    pass


class StreamLimitExceeded(RuntimeError):
    pass


@dataclass
class StreamReceipt:
    chunks: int = 0
    response_bytes: int = 0
    payload_bytes: int = 0
    first_chunk_seconds: object = None
    finish_reason: object = None
    usage: object = None
    completed: bool = False

    def metrics(self):
        return {"enabled": True, "chunks": self.chunks, "response_bytes": self.response_bytes,
                "payload_bytes": self.payload_bytes,
                "first_chunk_seconds": self.first_chunk_seconds, "finish_reason": self.finish_reason,
                "usage_received": self.usage is not None, "completed": self.completed}


class WireEvents:
    """Observe SDK events before pinned LiteLLM can synthesize finish/usage.

    OpenAI-compatible streams are SDK objects, not parsed SSE strings here.
    The decorator preserves the original iterator and close operation.
    """

    def __init__(self, stream, receipt, started, output_cap):
        self.stream, self.receipt, self.started = stream, receipt, started
        self.iterator = iter(stream)
        self.byte_cap = max(1024 * 1024, output_cap * 64)
        # Repeated SDK envelopes/metadata are not generated model content. Bound
        # both separately so a legitimate token-per-event stream fits its cap.
        self.wire_cap = max(8 * 1024 * 1024, output_cap * 2048)
        self.chunk_cap = max(2048, output_cap * 8)

    def __iter__(self):
        return self

    def __next__(self):
        chunk = next(self.iterator)
        check_cancelled()
        receipt = self.receipt
        if receipt.first_chunk_seconds is None:
            receipt.first_chunk_seconds = round(time.monotonic() - self.started, 6)
        receipt.chunks += 1
        raw = chunk.model_dump() if hasattr(chunk, 'model_dump') else chunk
        receipt.response_bytes += len(json.dumps(raw, ensure_ascii=False).encode('utf-8'))
        deltas = [field(c, 'delta', {}) for c in field(chunk, 'choices', [])]
        for delta in deltas:
            value = delta.model_dump(exclude_none=True) if hasattr(delta, 'model_dump') else delta
            receipt.payload_bytes += len(json.dumps(value, ensure_ascii=False).encode('utf-8'))
        if (receipt.chunks > self.chunk_cap or receipt.payload_bytes > self.byte_cap
                or receipt.response_bytes > self.wire_cap):
            raise StreamLimitExceeded('Provider stream exceeded bounded response storage')
        if field(chunk, 'usage') is not None:
            receipt.usage = field(chunk, 'usage')
        for choice in field(chunk, 'choices', []):
            if field(choice, 'index') != 0:
                raise IncompleteStream('Unexpected multiple response choices')
            finish = field(choice, 'finish_reason')
            if finish is not None:
                if receipt.finish_reason is not None and receipt.finish_reason != finish:
                    raise IncompleteStream('Conflicting terminal response events')
                receipt.finish_reason = finish
        return chunk

    def close(self):
        if hasattr(self.stream, 'close'):
            self.stream.close()


class NativeStream:
    """Close the SDK socket as well as disposing the LiteLLM wrapper."""

    def __init__(self, wrapper, wire):
        self.wrapper, self.wire = wrapper, wire

    def __iter__(self):
        return iter(self.wrapper)

    def close(self):
        self.wire.close()


def collect_response(factory, messages, timeout, output_cap, receipt):
    import litellm

    started = time.monotonic()
    def open_stream():
        wrapper = factory()
        # LiteLLM 1.82.6 is pinned in requirements. Its outer wrapper fabricates
        # terminal chunks/usage on clean EOF, so only raw SDK events can release
        # a reservation or authorize returning an executable assistant message.
        if not hasattr(wrapper, 'fetch_sync_stream'):
            if hasattr(wrapper, 'close'):
                wrapper.close()
            raise IncompleteStream('Native stream boundary unavailable')
        try:
            raw = wrapper.fetch_sync_stream()
            wire = WireEvents(raw, receipt, started, output_cap)
            wrapper.completion_stream = wire
            return NativeStream(wrapper, wire)
        except BaseException:
            raw = getattr(wrapper, 'completion_stream', None)
            if hasattr(raw, 'close'):
                raw.close()
            raise
    stream = interruptible_call(open_stream, timeout=timeout)
    chunks = []
    remaining = max(.001, timeout - (time.monotonic() - started))
    reader = interruptible_stream(stream, timeout=remaining)
    try:
        for chunk in reader:
            check_cancelled()
            chunks.append(chunk)
    finally:
        reader.close()
    check_cancelled()
    if not chunks or receipt.finish_reason is None:
        raise IncompleteStream('Stream ended without a terminal response; no tools executed')
    response = litellm.stream_chunk_builder(chunks, messages=messages)
    if response is None or not response.choices:
        raise IncompleteStream('No complete assistant response; no tools executed')
    # The builder can prefer an earlier tool fragment's finish state. Preserve
    # the provider's actual terminal reason, especially length/content_filter.
    response.choices[0].finish_reason = receipt.finish_reason
    # LiteLLM may synthesize usage when the provider omits it. Never use that
    # estimate to release a hard-budget reservation or report measured tokens.
    response.usage = receipt.usage
    receipt.completed = True
    return response
