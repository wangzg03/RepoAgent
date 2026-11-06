"""Public development fixtures; no user memories or benchmark answers."""

import json
import sqlite3
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from test_harness import ScriptedProvider
from test_loop import IMAGE

from corecoder_mini.context import estimate_tokens
from corecoder_mini.context_view import ContextView
from corecoder_mini.engine import build_agent
from corecoder_mini.events import Events
from corecoder_mini.legacy_memory import MemoryStore
from corecoder_mini.memory_embeddings import OllamaMemoryEmbedder, local_endpoint
from corecoder_mini.project_memory import HybridMemoryStore
from corecoder_mini.runtime import Cancelled, local
from corecoder_mini.state_tools import ProjectMemoryTool, WorkingState
from corecoder_mini.supervisor import DEFAULTS, validate_extensions
from corecoder_mini.tools.registry import ToolRegistry


@pytest.fixture(autouse=True)
def clear_runtime():
    vars(local).clear()
    yield
    vars(local).clear()


class SemanticFixture:
    """Known concept vectors test wiring, not a model-quality measurement."""
    identity = "fixture:semantic@v1"

    def __init__(self):
        self.calls = 0

    def __call__(self, text):
        self.calls += 1
        return [1., 0.] if any(word in text.lower() for word in ["rollback", "restore", "撤销"]) else [0., 1.]


def test_semantic_only_hit_and_multi_channel_scores_survive_restart(tmp_path):
    store = HybridMemoryStore(tmp_path / "memory.db", "one", SemanticFixture())
    identity = store.save("Restore the previous revision", key="recovery")
    store.save("Use the virtual environment for checks", key="tests")
    reopened = HybridMemoryStore(store.path, "one", SemanticFixture())
    hits = reopened.search("撤销")
    assert hits[0]["id"] == identity and hits[0]["channels"] == {"vector": 1.0}
    hits = reopened.search("restore")
    assert set(hits[0]["channels"]) == {"vector", "bm25", "keyword"}
    assert reopened.last_trace["records_scanned"] == 2


def test_real_bm25_length_normalization_and_rare_terms(tmp_path):
    store = HybridMemoryStore(tmp_path / "memory.db", "one")
    short = store.save("storage lease restore", key="short")
    long = store.save("storage lease restore " + "unrelated " * 200, key="long")
    store.save("storage cleanup", key="common")
    hits = store.search("storage lease")
    scores = {r["id"]: r["channels"]["bm25"] for r in hits}
    assert scores[short] > scores[long] > 0
    assert hits[0]["id"] == short
    assert store.search('" OR nonexistent -- lease')[0]["id"] == short


def test_identifier_and_chinese_queries_work_without_embedding(tmp_path):
    store = HybridMemoryStore(tmp_path / "memory.db", "one")
    a = store.save("Call restoreWorkspace when interrupted; retain storage_lease.")
    b = store.save("中断恢复需要保留任务标识")
    assert store.search("restoreWorkspace")[0]["id"] == a
    assert store.search("storage_lease")[0]["id"] == a
    assert store.search("中断恢复")[0]["id"] == b
    assert "vector" not in store.search("storage_lease")[0]["channels"]


def test_scope_expiry_and_global_preferences_never_enter_project_retrieval(tmp_path):
    path = tmp_path / "memory.db"
    legacy = MemoryStore(path, "one")
    own = legacy.save("restore own revision")
    expired = legacy.save("restore expired", ttl_days=1)
    preference = legacy.save("restore global preference", kind="preference")
    foreign = MemoryStore(path, "two").save("restore foreign revision")
    with legacy._db() as db:
        db.execute("UPDATE memories SET expires=0 WHERE id=?", (expired,))
    store = HybridMemoryStore(path, "one", SemanticFixture())
    assert [r["id"] for r in store.search("restore")] == [own]
    assert store.forget(foreign) == store.forget(preference) == 0
    assert store.reindex(limit=1)["updated"] == 1
    with store._db() as db:
        assert db.execute("SELECT COUNT(*) FROM memories WHERE vector IS NOT NULL").fetchone()[0] == 1


def test_legacy_notes_migrate_in_bounded_batches_and_model_digest_change_is_detected(tmp_path):
    path = tmp_path / "memory.db"
    old = MemoryStore(path, "one")
    first = old.save("Restore one revision")
    old.save("Restore another revision")
    provider = SemanticFixture()
    store = HybridMemoryStore(path, "one", provider)
    assert store.search("restore")[0]["channels"].keys() == {"bm25", "keyword"}
    result = store.reindex(limit=1)
    assert result["updated"] == result["remaining_in_window"] == 1
    assert store.reindex(limit=1)["remaining_in_window"] == 0
    assert store.search("撤销")[0]["channels"] == {"vector": 1.0}
    provider.identity = "fixture:semantic@v2"
    assert store.search("撤销") == []
    assert store.last_trace["incompatible_vectors"] == 2
    assert store.reindex(limit=2)["updated"] == 2
    assert first in [row["id"] for row in store.search("撤销")]


def test_reindex_repairs_corrupt_vectors_even_when_the_model_identity_matches(tmp_path):
    store = HybridMemoryStore(tmp_path / "memory.db", "one", SemanticFixture())
    identity = store.save("restore a revision")
    with store._db() as db:
        db.execute("UPDATE memories SET vector='[0.0]' WHERE id=?", (identity,))
    assert store.search("撤销") == []
    assert store.reindex(limit=1)["updated"] == 1
    assert store.search("撤销")[0]["id"] == identity


def test_missing_optional_fts_build_keeps_vector_and_keyword_fallback(tmp_path, monkeypatch):
    store = HybridMemoryStore(tmp_path / "memory.db", "one", SemanticFixture())
    store.save("restore a revision")
    real = sqlite3.connect
    def connect(path, *args, **kwargs):
        if path == ":memory:":
            raise sqlite3.OperationalError("no such module fts5")
        return real(path, *args, **kwargs)
    monkeypatch.setattr(sqlite3, "connect", connect)
    hits = store.search("restore")
    assert set(hits[0]["channels"]) == {"vector", "keyword"}
    assert store.last_trace["bm25_error"] == "OperationalError"


def test_temporary_bm25_connections_release_on_every_search(tmp_path, monkeypatch):
    store = HybridMemoryStore(tmp_path / "memory.db", "one")
    store.save("restore a revision")
    real, closed = sqlite3.connect, []
    class Connection(sqlite3.Connection):
        def close(self):
            closed.append(True)
            return super().close()
    def connect(path, *args, **kwargs):
        return real(path, *args, **kwargs, **({"factory": Connection} if path == ":memory:" else {}))
    monkeypatch.setattr(sqlite3, "connect", connect)
    for _ in range(3):
        assert store.search("restore")
    assert len(closed) == 3


@pytest.mark.parametrize("bad", [[float("nan")], [0.], [], [True], [float("inf")]])
def test_bad_vectors_degrade_to_lexical_with_safe_diagnosis(tmp_path, bad):
    provider = SemanticFixture()
    provider.__class__ = type("BrokenVector", (), {"identity": "bad", "__call__": lambda self, text: bad})
    store = HybridMemoryStore(tmp_path / "memory.db", "one", provider)
    store.save("restore documented revision")
    hits = store.search("restore")
    assert hits and hits[0]["embedding_error"] == "ValueError"
    assert set(hits[0]["channels"]) == {"bm25", "keyword"}


def test_failed_embedding_is_not_retried_per_note_and_private_error_is_not_exposed(tmp_path):
    class Offline(SemanticFixture):
        def __call__(self, text):
            self.calls += 1
            raise OSError("PRIVATE_FIXTURE_server_body")
    provider = Offline()
    store = HybridMemoryStore(tmp_path / "memory.db", "one", provider)
    store.save("restore revision")
    for _ in range(4):
        assert store.search("restore")
    assert provider.calls == 1 and store.status()["retry_backoff"]
    assert "PRIVATE_FIXTURE" not in json.dumps(store.status())


def test_reindex_cannot_overwrite_concurrently_replaced_note(tmp_path):
    path = tmp_path / "memory.db"
    legacy = MemoryStore(path, "one")
    identity = legacy.save("old restore note")
    class Concurrent(SemanticFixture):
        def __call__(self, text):
            if text == "old restore note":
                with legacy._db() as db:
                    db.execute("UPDATE memories SET text='replacement' WHERE id=?", (identity,))
            return super().__call__(text)
    store = HybridMemoryStore(path, "one", Concurrent())
    assert store.reindex()["updated"] == 0
    with store._db() as db:
        assert db.execute("SELECT text,vector FROM memories WHERE id=?", (identity,)).fetchone() == ("replacement", None)


def test_cancelled_embedding_never_commits_and_is_not_lexical_success(tmp_path):
    class Cancelling(SemanticFixture):
        def __call__(self, text):
            local.cancel.set()
            return super().__call__(text)
    store = HybridMemoryStore(tmp_path / "memory.db", "one", Cancelling())
    local.cancel = threading.Event()
    with pytest.raises(Cancelled):
        store.save("restore revision")
    local.cancel.clear()
    assert store.status()["project_records"] == 0


def test_tool_registry_status_reindex_and_benchmark_no_store_no_transport(tmp_path, monkeypatch):
    tool = ProjectMemoryTool(tmp_path / "memory.db", "one")
    registry = ToolRegistry()
    registry.register(tool)
    assert {"status", "reindex"} <= set(tool.parameters["properties"]["action"]["enum"])
    identity = json.loads(tool.execute("save", text="Use isolated checks"))["identity"]
    assert json.loads(tool.execute("status"))["project_records"] == 1
    assert json.loads(tool.execute("reindex"))["embedding_error"] == "embedding_off"
    assert json.loads(tool.execute("read", identity=identity))["text"] == "Use isolated checks"
    with pytest.raises(ValueError, match="credential"):
        tool.execute("save", text="sk-" + "PUBLIC_FIXTURE_" * 2)
    monkeypatch.setattr(urllib.request, "build_opener", lambda *args: pytest.fail("benchmark transport"))
    disabled = ProjectMemoryTool(tmp_path / "must-not-exist.db", "one", enabled=False,
                                 settings={"memory_embedding_url": "https://invalid"})
    for action in ["search", "save", "status", "reindex", "read", "forget"]:
        assert "quarantined" in disabled.execute(action)
    assert disabled.context("goal") == "" and not disabled.store and not (tmp_path / "must-not-exist.db").exists()


def test_startup_recall_is_cached_bounded_and_not_canonical_or_instructions(tmp_path):
    tool = ProjectMemoryTool(tmp_path / "memory.db", "one")
    tool.store.save("restore " + "public fact " * 300)
    tool.store.provider = SemanticFixture()
    view = ContextView(tmp_path / "agent", WorkingState(tmp_path / "agent"), Events(tmp_path / "agent"))
    view.goal_provider = lambda: "restore"
    view.project_memory = tool.context
    messages = [{"role": "user", "content": "restore"}]
    output, receipt = view.project(messages, 4000)
    assert receipt["project_memory_included"] and "possibly stale data" in output[-1]["content"]
    assert len(tool.context("restore")) < 1100
    view.project(messages, 4000)
    assert tool.store.provider.calls == 1
    assert "public fact" not in (tmp_path / "agent/canonical-messages.json").read_text()
    # Old notes are expendable when this task fills the provider capacity.
    tool.context_cache.clear()
    output, receipt = view.project(messages, 150)
    assert not receipt["project_memory_included"] and estimate_tokens(output) <= 150


def test_runtime_registers_local_hybrid_memory_but_children_and_benchmarks_do_not_open_it(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    for kind, depth in [("user", 0), ("benchmark", 0), ("user", 1)]:
        directory = tmp_path / f"{kind}-{depth}"
        agent = build_agent(ScriptedProvider([]), workspace, directory, IMAGE,
                            {**DEFAULTS, "max_subagents": 0, "harness_tools": ["memory"], "task_kind": kind}, depth=depth)
        store = agent.env.registry.tools["memory"].store
        if kind == "benchmark" or depth:
            assert store is None and not (directory / "project-memory.sqlite3").exists()
        else:
            assert isinstance(store.provider, OllamaMemoryEmbedder)


@pytest.mark.parametrize("url", ["https://example.com:443", "http://192.168.0.2:11434", "http://localhost:11434/path",
                                   "http://user:pass@localhost:11434", "http://localhost:11434?secret=x"])
def test_memory_cannot_be_configured_to_send_notes_to_remote_endpoints(url):
    with pytest.raises(ValueError, match="loopback"):
        local_endpoint(url)


@pytest.mark.parametrize("options", [{"memory_embedding": "unknown"}, {"memory_embedding_seconds": 0},
                                     {"memory_startup_recall": 1}, {"memory_embedding_model": "x:cloud"}])
def test_bad_memory_settings_fail_before_inference(options):
    with pytest.raises(ValueError):
        validate_extensions(options)


def test_local_embedding_transport_pins_digest_normalizes_and_respects_timeout():
    calls = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            calls.append(self.path)
            self.send({"models": [{"name": "fixture:latest", "digest": "fixture-digest"}]})

        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(data)
            self.send({"embeddings": [[3., 4.]]})

        def send(self, data):
            body = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        provider = OllamaMemoryEmbedder("fixture:latest", f"http://127.0.0.1:{server.server_port}", 2)
        assert provider("public development text") == [.6, .8]
        assert provider.identity.endswith("fixture:latest@fixture-digest")
        assert calls[1]["truncate"] is False and calls[1]["options"]["num_thread"] == 2
        assert calls[0] == calls[-1] == "/api/tags"
        local.deadline = time.monotonic() - .1
        with pytest.raises(Cancelled):
            provider("not sent")
        assert len(calls) == 3
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_cancel_during_actual_local_transport_returns_promptly_and_late_result_cannot_save(tmp_path):
    entered, release = threading.Event(), threading.Event()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send({"models": [{"name": "fixture:latest", "digest": "fixture-digest"}]})

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            entered.set()
            release.wait(3)
            self.send({"embeddings": [[3., 4.]]})

        def send(self, data):
            body = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    provider = OllamaMemoryEmbedder("fixture:latest", f"http://127.0.0.1:{server.server_port}", 2)
    store = HybridMemoryStore(tmp_path / "memory.db", "one", provider)
    local.cancel = threading.Event()
    cancel = local.cancel
    def stop():
        if entered.wait(2):
            cancel.set()
    stopper = threading.Thread(target=stop)
    stopper.start()
    try:
        started = time.monotonic()
        with pytest.raises(Cancelled):
            store.save("public fixture note")
        assert time.monotonic() - started < 1.5
        assert provider.identity == "unresolved_local_embedding"
    finally:
        release.set()
        stopper.join(timeout=3)
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
    local.cancel.clear()
    assert store.status()["project_records"] == 0
