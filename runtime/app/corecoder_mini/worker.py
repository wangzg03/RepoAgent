"""Owned child process; the supervisor, not this process, owns final delivery."""

import argparse
import os
import threading
import time
from pathlib import Path

from .budget import Budget
from .delivery import read_json
from .engine import build_agent
from .native_model import NativeProvider
from .runtime import local
from .task_state import atomic_json


def create_provider(model, settings, budget):
    provider = model.get("provider", "ollama")
    key = os.environ.get(model["api_key_env"]) if provider == "openai" else "ollama-local"
    if not key:
        raise ValueError("credential_environment_missing")
    env_name = model["api_key_env"] if provider == "openai" else "CORECODER_LOCAL_DUMMY_KEY"
    if provider != "openai":
        os.environ[env_name] = key
    return NativeProvider(model=model["name"], api_key_env=env_name,
                          base_url=model.get("base_url", "http://127.0.0.1:11434/v1"), budget=budget,
                          pricing=tuple(model.get("pricing_usd_per_million", [0, 0])), settings=settings)


def run(directory):
    directory = Path(directory).resolve()
    frozen = read_json(directory / "frozen.json")
    settings = frozen["settings"]
    cancel, finished = threading.Event(), threading.Event()
    local.cancel = cancel
    local.deadline = time.monotonic() + settings["agent_seconds"]
    local.root = directory / "workspace"
    local.cwd = local.root

    def observe_stop():
        while not finished.wait(.2):
            if (directory / "STOP").exists():
                cancel.set()
                return

    threading.Thread(target=observe_stop, daemon=True).start()
    budget = Budget(settings["dollar_limit"], token_limit=settings["token_limit"],
                    request_limit=settings["request_limit"])
    # Secrets are read in the worker, never written into the frozen config or switched to a fallback.
    llm = create_provider(frozen["model"], settings, budget)
    agent = build_agent(llm, directory / "workspace", directory / "agent", frozen["image_id"],
                        settings, frozen["job_id"])
    exit_code = 1
    try:
        agent.working_state.bind_goal(frozen["public_task"])
        result = agent.run(frozen["public_task"])
        atomic_json(directory / "worker-result.json", {"status": "exited", **result})
        exit_code = 0 if result.get("exit_status") == "Submitted" else 1
    except BaseException as exc:  # noqa: BLE001 - process boundary records interruption before exit
        atomic_json(directory / "worker-result.json", {"status": "failed", "error_type": type(exc).__name__,
                                                       "provider_diagnostic": getattr(exc, "diagnostic", None)})
    finally:
        finished.set()
        try:
            agent.env.cleanup()
        finally:
            agent.save(directory / "agent/trajectory.json")
            atomic_json(directory / "budget.json", {"requests": budget.requests,
                        "retry_attempts": budget.retry_attempts,
                        "reported_prompt_tokens": budget.prompt_tokens, "reported_completion_tokens": budget.completion_tokens,
                        "reported_cached_prompt_tokens": budget.cached_prompt_tokens,
                        "unknown_usage_requests": budget.unknown_usage_requests,
                        "known_usage_usd_estimate": budget.known_usage_cost,
                        "conservative_accounted_tokens": budget.tokens_spent, "usd_estimate": budget.spent,
                        "is_invoice": False})
    return exit_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    raise SystemExit(run(parser.parse_args().directory))
