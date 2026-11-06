"""Shared, atomic dollar reservations for parent, children, retries and compaction."""

import json
import math
import threading
import uuid
from contextlib import contextmanager
from contextvars import ContextVar

from .accounting import usage_charge, validate_rates


class BudgetExceeded(RuntimeError):
    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details or {}


class DeliveryReserved(BudgetExceeded):
    """Investigation retries must leave the child's last HTTP attempt for delivery."""


class ChildBudget:
    """Count child HTTP attempts locally while charging the same parent ledger."""

    def __init__(self, shared, request_limit=3, reserve_delivery=False):
        self.shared, self.request_limit = shared, request_limit
        self.requests = 0
        self._child_lock = threading.Lock()
        self.reserve_delivery = reserve_delivery
        self.delivering = False

    def enter_delivery(self):
        with self._child_lock:
            self.delivering = True

    def __getattr__(self, name):
        return getattr(self.shared, name)

    def reserve(self, params, rates):
        with self._child_lock:
            if self.requests >= self.request_limit:
                raise BudgetExceeded("Child provider attempt limit exhausted; main must take over")
            if self.reserve_delivery and not self.delivering and self.requests >= self.request_limit - 1:
                raise DeliveryReserved("Final child HTTP attempt reserved for delivery")
            reservation = self.shared.reserve(params, rates)
            self.requests += 1
            return reservation

    def settle(self, amount, usage, rates):
        return self.shared.settle(amount, usage, rates)


class Reservation(float):
    def __new__(cls, amount, tokens):
        value = super().__new__(cls, amount)
        value.tokens = tokens
        return value


class Budget:
    def __init__(self, limit, token_limit=None, request_limit=None):
        if not math.isfinite(limit) or limit < 0:
            raise ValueError("Dollar budget must be finite and non-negative")
        self.limit = limit
        if token_limit is not None and (not isinstance(token_limit, int) or isinstance(token_limit, bool) or token_limit < 1):
            raise ValueError("Token budget must be a positive integer or None")
        self.token_limit = token_limit
        if request_limit is not None and (not isinstance(request_limit, int) or isinstance(request_limit, bool) or request_limit < 1):
            raise ValueError("Provider request budget must be a positive integer or None")
        self.request_limit = request_limit
        self.tokens_spent = self.tokens_reserved = self.requests = 0
        self.retry_attempts = 0
        self.spent = 0.0
        self.reserved = 0.0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cached_prompt_tokens = 0
        self.unknown_usage_requests = 0
        self.known_usage_cost = 0.0
        self.blocked = False
        self._lock = threading.Lock()
        self._holds = {}
        self._claim = ContextVar("corecoder_budget_claim", default=None)
        self._known_reservations = set()

    def retries_remaining(self, limit):
        with self._lock:
            return max(0, limit - self.retry_attempts)

    def claim_retry(self, limit):
        with self._lock:
            if self.retry_attempts >= limit:
                raise BudgetExceeded("Shared provider retry limit exhausted")
            self.retry_attempts += 1

    def hold_delivery(self, key):
        """Leave one bounded review attempt inside the existing task caps."""
        with self._lock:
            if key in self._holds or self.blocked:
                return False
            dollars, tokens, requests = self._held(exclude_claim=False)
            remaining = self.limit - self.spent - self.reserved - dollars
            remaining_tokens = (self.token_limit - self.tokens_spent - self.tokens_reserved - tokens
                                if self.token_limit is not None else None)
            if (remaining <= 0 or remaining_tokens is not None and remaining_tokens < 8192
                    or self.request_limit is not None and self.request_limit - self.requests - requests < 3):
                return False
            self._holds[key] = (remaining / 4, min(65536, remaining_tokens // 4) if remaining_tokens is not None else 0, 1)
            return True

    def release_delivery(self, key):
        with self._lock:
            self._holds.pop(key, None)

    @contextmanager
    def delivery(self, key):
        token = self._claim.set(key)
        try:
            yield
        finally:
            self._claim.reset(token)

    def _held(self, exclude_claim=True):
        values = [value for key, value in self._holds.items() if not exclude_claim or key != self._claim.get()]
        return tuple(sum(value[i] for value in values) for i in range(3))

    @staticmethod
    def quote(params, rates):
        try:
            rates = validate_rates(rates)
        except ValueError as exc:
            raise BudgetExceeded("Hard budget requires explicit valid pricing for every model") from exc
        output = params.get("max_completion_tokens", params.get("max_tokens"))
        if not isinstance(output, int) or output <= 0:
            raise BudgetExceeded("Hard budget requires a positive output token cap")
        # UTF-8 bytes plus generous per-message/schema overhead is deliberately
        # conservative for byte-based tokenizers; custom providers must honor caps.
        payload = {k: params[k] for k in ("messages", "tools") if k in params}
        input_cap = len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) + 4096
        # Never assume a future cache hit in the hard preflight reservation.
        amount = (input_cap * max(rates[0], rates[-1] if len(rates) == 3 else rates[0]) + output * rates[1]) / 1_000_000
        tokens = input_cap + output
        return Reservation(amount, tokens)

    def _admission(self, amount):
        """Called under the ledger lock; previews and reservations use identical rules."""
        dollars, tokens, requests = self._held()
        blockers = []
        if self.blocked:
            blockers.append("provider_bound_violation")
        if self.request_limit is not None and self.requests + requests >= self.request_limit:
            blockers.append("requests")
        if self.spent + self.reserved + amount + dollars > self.limit + 1e-12:
            blockers.append("usd")
        if self.token_limit is not None and self.tokens_spent + self.tokens_reserved + amount.tokens + tokens > self.token_limit:
            blockers.append("tokens")
        return {"allowed": not blockers, "blocking_limits": blockers,
                "required_usd": float(amount), "required_tokens": amount.tokens,
                "available_usd": max(0, self.limit - self.spent - self.reserved - dollars),
                "available_tokens": None if self.token_limit is None else max(0, self.token_limit - self.tokens_spent - self.tokens_reserved - tokens),
                "available_requests": None if self.request_limit is None else max(0, self.request_limit - self.requests - requests),
                "held_usd": dollars, "held_tokens": tokens, "held_requests": requests}

    def admission(self, params, rates):
        """Safe numeric diagnosis without a reservation, transport or prompt logging."""
        amount = self.quote(params, rates)
        with self._lock:
            return self._admission(amount)

    def fits(self, params, rates):
        return self.admission(params, rates)["allowed"]

    def reserve(self, params, rates):
        amount = self.quote(params, rates)
        tokens = amount.tokens
        with self._lock:
            diagnosis = self._admission(amount)
            if not diagnosis["allowed"]:
                labels = {"provider_bound_violation": "Provider violated a token bound; budget is locked",
                          "requests": "Shared provider request budget exhausted",
                          "usd": f"USD budget exhausted: limit=${self.limit:.6f}, spent=${self.spent:.6f}",
                          "tokens": f"Shared token budget exhausted: limit={self.token_limit}, accounted={self.tokens_spent}"}
                raise BudgetExceeded("; ".join(labels[key] for key in diagnosis["blocking_limits"]), diagnosis)
            self.reserved += amount
            self.tokens_reserved += tokens
            self.requests += 1
            self._holds.pop(self._claim.get(), None)  # Atomically convert the hold to a real reservation.
            reservation = Reservation(amount, tokens)
            reservation.identity = uuid.uuid4().hex
            self._known_reservations.add(reservation.identity)
        return reservation

    def settle(self, amount, usage, rates):
        cost = amount  # missing usage / interrupted requests consume the reservation
        bound = getattr(amount, "tokens", 0)
        tokens = bound
        prompt = completion = 0
        known = usage_charge(usage, rates)
        if known is not None:
            cost = known["cost_usd"]
            prompt, completion, tokens = known["prompt_tokens"], known["completion_tokens"], known["tokens"]
        with self._lock:
            self.reserved -= amount
            self.spent += cost
            self.tokens_reserved -= bound
            self.tokens_spent += tokens
            self.prompt_tokens += prompt
            self.completion_tokens += completion
            self.cached_prompt_tokens += (known or {}).get("cached_prompt_tokens") or 0
            self.unknown_usage_requests += known is None
            self.known_usage_cost += cost if known is not None else 0
            violated = cost > amount + 1e-12 or self.token_limit is not None and tokens > bound
            if violated:
                self.blocked = True
            amount.receipt = {"cost_usd": float(cost), "tokens": tokens,
                              "prompt_tokens": prompt, "completion_tokens": completion,
                              "usage_complete": False, "cached_prompt_tokens": None,
                              "cost_basis": "unknown_usage_reservation", "is_invoice": False,
                              **(known or {}), "bound_violated": violated}
        if violated:
            raise BudgetExceeded("Provider exceeded the reserved token bound; further requests stopped")

    def reconcile(self, entries):
        """Restore task receipts once, charging an unresolved submission at its full cap."""
        with self._lock:
            for item in entries:
                identity = item["reservation_id"]
                if identity in self._known_reservations:
                    continue
                self._known_reservations.add(identity)
                receipt = item.get("receipt") or {"cost_usd": item["reserved_usd"], "tokens": item["reserved_tokens"]}
                self.requests += 1
                self.spent += receipt["cost_usd"]
                self.tokens_spent += receipt["tokens"]
                self.prompt_tokens += receipt.get("prompt_tokens", 0)
                self.completion_tokens += receipt.get("completion_tokens", 0)
                self.blocked |= receipt.get("bound_violated", False)
