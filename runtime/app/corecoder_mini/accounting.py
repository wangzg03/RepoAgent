"""Configured tariff estimates, never invoices. Missing usage stays conservative."""

import math


def validate_rates(rates):
    # Backward compatible: [uncached input, output, optional cached input], USD/M.
    if (not isinstance(rates, (list, tuple)) or len(rates) not in {2, 3}
            or any(type(r) not in {int, float} or not math.isfinite(r) or r < 0 for r in rates)):
        raise ValueError("pricing_requires_input_output_and_optional_cached_input")
    return tuple(rates)


def field(value, name, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def usage_charge(usage, rates):
    """Return a verifiable tariff calculation or None for unknown/malformed usage."""
    rates = validate_rates(rates)
    prompt, completion = field(usage, "prompt_tokens"), field(usage, "completion_tokens")
    if any(type(n) is not int or n < 0 for n in (prompt, completion)):
        return None
    cached = field(field(usage, "prompt_tokens_details"), "cached_tokens")
    if cached is None:
        cached = field(usage, "prompt_cache_hit_tokens")
    if type(cached) is not int or not 0 <= cached <= prompt:
        cached = None
    cache_rate = rates[2] if len(rates) == 3 else rates[0]
    charged_cache = cached if cached is not None else 0
    cost = ((prompt - charged_cache) * rates[0] + charged_cache * cache_rate + completion * rates[1]) / 1_000_000
    return {"cost_usd": cost, "prompt_tokens": prompt, "completion_tokens": completion,
            "tokens": prompt + completion, "cached_prompt_tokens": cached,
            "usage_complete": True, "cache_usage_complete": cached is not None,
            "cost_basis": ("cache_aware_tariff_estimate" if len(rates) == 3 and cached is not None else
                           "uncached_upper_estimate" if len(rates) == 3 else "two_rate_estimate"),
            "is_invoice": False}
