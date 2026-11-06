"""Bounded recovery and evidence using public localhost fixtures, no paid API."""

import json
import threading

import httpx
import pytest
from test_native_upstream import local_provider, make_agent, response

from corecoder_mini.budget import Budget, BudgetExceeded
from corecoder_mini.delivery import read_json
from corecoder_mini.native_model import ProviderFailure
from corecoder_mini.provider_diagnostics import error_diagnostics, retry_decision
from corecoder_mini.runtime import Cancelled, local
from corecoder_mini.supervisor import validate_extensions


@pytest.fixture(autouse=True)
def state():
    vars(local).clear()
    yield
    vars(local).clear()


def test_recovery_keeps_payload_and_accounts_each_attempt_once(tmp_path, monkeypatch):
    def reply(data, number):
        if number == 1:
            return 503, {"error": {"type": "server_error", "message": "fixture upstream unavailable"}}, {"x-request-id": "fixture-error-001"}
        return 200, response([("bash", {"command": "echo recovered"})])
    with local_provider(reply) as (url, requests):
        agent = make_agent(tmp_path, monkeypatch, url, provider_attempts=3)
        message = agent.model.query([{"role": "user", "content": "Public fixture"}])
        assert requests[0] == requests[1] and len(requests) == 2
        assert agent.model.provider.budget.retry_attempts == 1
        first, second = [read_json(tmp_path / "agent/requests" / name) for name in ["0001.json", "0002.json"]]
        assert first["diagnostic"]["http_status"] == 503
        assert first["diagnostic"]["request_id"] == "fixture-error-001"
        assert first["diagnostic"]["category"] == "server" and first["retry_decision"]["will_retry"]
        assert first["logical_call"] == second["logical_call"]
        assert first["payload_sha256"] == second["payload_sha256"]
        assert [first["attempt"], second["attempt"]] == [1, 2]
        assert message["extra"]["cost"] == pytest.approx(first["cost"] + second["cost"])
        assert not (tmp_path / "agent/actions").exists()  # Only one final action may be delivered later.


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_permanent_provider_failure_stops_without_retry(tmp_path, monkeypatch, status):
    with local_provider(lambda *args: (status, {"error": {"message": "invalid request fixture", "type": "invalid_request_error"}})) as (url, requests):
        agent = make_agent(tmp_path, monkeypatch, url, provider_attempts=3)
        with pytest.raises(ProviderFailure):
            agent.model.query([{"role": "user", "content": "Public fixture"}])
        assert len(requests) == 1 and agent.model.provider.budget.retry_attempts == 0
        receipt = read_json(tmp_path / "agent/requests/0001.json")
        assert receipt["diagnostic"]["http_status"] == status
        assert receipt["retry_decision"]["reason"] == "nonrecoverable"


def test_task_retry_cap_is_shared_across_logical_queries(tmp_path, monkeypatch):
    monkeypatch.setattr("corecoder_mini.native_model.retry_delay", lambda seconds: None)
    with local_provider(lambda *args: (503, {"error": {"message": "fixture unavailable"}})) as (url, requests):
        agent = make_agent(tmp_path, monkeypatch, url, provider_attempts=3, provider_retry_limit=1)
        for _ in range(2):
            with pytest.raises(ProviderFailure):
                agent.model.query([{"role": "user", "content": "Public fixture"}])
        assert len(requests) == 3 and agent.model.provider.budget.retry_attempts == 1
        assert read_json(tmp_path / "agent/requests/0003.json")["retry_decision"]["reason"] == "task_retry_limit"


def test_retry_claim_is_atomic_across_children():
    budget = Budget(1)
    barrier = threading.Barrier(2)
    results = []
    def claim():
        barrier.wait(timeout=3)
        try:
            budget.claim_retry(1)
            results.append("claimed")
        except BudgetExceeded:
            results.append("blocked")
    threads = [threading.Thread(target=claim) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)
    assert sorted(results) == ["blocked", "claimed"] and budget.retry_attempts == 1


def test_budget_blocks_retry_before_next_http(tmp_path, monkeypatch):
    with local_provider(lambda *args: (503, {"error": {"message": "fixture unavailable"}})) as (url, requests):
        agent = make_agent(tmp_path, monkeypatch, url, budget=Budget(.1, 100000, 1), provider_attempts=3)
        with pytest.raises(ProviderFailure):
            agent.model.query([{"role": "user", "content": "Public fixture"}])
        assert len(requests) == 1 and agent.model.provider.budget.retry_attempts == 0
        assert read_json(tmp_path / "agent/requests/0001.json")["retry_decision"]["reason"] == "shared_budget"


def test_provider_bound_violation_persists_usage_and_stops_tools(tmp_path, monkeypatch):
    def reply(*args):
        body = response([("bash", {"command": "echo must-not-run"})])
        body["usage"]["prompt_tokens"] = 999999
        return 200, body
    with local_provider(reply) as (url, requests):
        agent = make_agent(tmp_path, monkeypatch, url)
        with pytest.raises(BudgetExceeded):
            agent.model.query([{"role": "user", "content": "Public fixture"}])
        record = read_json(tmp_path / "agent/requests/0001.json")
        assert len(requests) == 1 and agent.model.provider.budget.blocked
        assert record["usage_complete"] and record["prompt_tokens"] == 999999
        assert record["diagnostic"]["stage"] == "budget_settlement"
        assert not (tmp_path / "agent/actions").exists()


def test_retry_after_is_recorded_and_long_wait_stops(tmp_path, monkeypatch):
    with local_provider(lambda *args: (429, {"error": {"message": "fixture rate limit"}}, {"Retry-After": "60"})) as (url, requests):
        agent = make_agent(tmp_path, monkeypatch, url, provider_attempts=3)
        with pytest.raises(ProviderFailure):
            agent.model.query([{"role": "user", "content": "Public fixture"}])
        receipt = read_json(tmp_path / "agent/requests/0001.json")
        assert len(requests) == 1 and receipt["diagnostic"]["retry_after_seconds"] == 60
        assert receipt["retry_decision"]["reason"] == "retry_after_too_long"
    decision = retry_decision({"category": "rate_limit", "retry_after_seconds": 3}, 1, 3, 3, None)
    assert decision["delay_seconds"] == 3


def test_cancel_during_retry_wait_never_resends_request(tmp_path, monkeypatch):
    event = threading.Event()
    timers = []
    def reply(*args):
        timer = threading.Timer(.15, event.set)
        timers.append(timer)
        timer.start()
        return 503, {"error": {"message": "fixture unavailable"}}
    with local_provider(reply) as (url, requests):
        agent = make_agent(tmp_path, monkeypatch, url, provider_attempts=3)
        local.cancel = event
        with pytest.raises(Cancelled):
            agent.model.query([{"role": "user", "content": "Public fixture"}])
        for timer in timers:
            timer.join(timeout=3)
        assert len(requests) == 1
        assert agent.model.provider.budget.retry_attempts == 0


def test_diagnostic_finds_wrapped_status_and_does_not_copy_private_echoes():
    secret = "private-fixture-credential"
    source = "private source content must never reach diagnostics"
    underlying = httpx.HTTPStatusError("fixture", request=httpx.Request("POST", "https://example.test/v1"),
                                     response=httpx.Response(502, headers={"x-request-id": "public-request-id", "content-type": "text/html"}))
    wrapper = RuntimeError(secret + source)
    wrapper.__cause__ = underlying
    result = error_diagnostics(wrapper, (secret,))
    assert result["http_status"] == 502 and result["category"] == "server"
    assert result["request_id"] == "public-request-id"
    assert secret not in json.dumps(result) and source not in json.dumps(result)
    assert result["raw_error_text_retained"] is False


def test_unknown_api_error_is_explicitly_bounded_and_deadline_can_stop_it():
    error = type("APIError", (RuntimeError,), {})("fixture unparsable upstream")
    result = error_diagnostics(error)
    assert result["category"] == "provider_api" and result["http_status"] is None
    assert retry_decision(result, 1, 3, 3, None)["will_retry"]
    assert retry_decision(result, 3, 3, 3, None)["reason"] == "attempt_limit"
    assert retry_decision(result, 1, 3, 3, 1)["reason"] == "deadline_reserve"


@pytest.mark.parametrize("settings", [{"provider_retry_limit": 4}, {"provider_retry_limit": True}, {"provider_attempts": 0}])
def test_invalid_retry_bounds_fail_before_transport(settings):
    with pytest.raises(ValueError):
        validate_extensions(settings)
