"""Reused R23 scratchpad and SQLite memory primitives; see REUSE_ASSESSMENT.md."""
import hashlib
import json
import math
import re
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path


class Scratchpad:
    def __init__(self):
        self.notes = {}
        self._lock = threading.RLock()

    def execute(self, action, key="", value=None):
        with self._lock:
            if action == "merge":
                if isinstance(value, str) and len(value) <= 32000:
                    try:
                        value = json.loads(value)
                    except ValueError:
                        pass
                if not isinstance(value, dict) or not all(isinstance(k, str) and k for k in value):
                    raise ValueError('merge needs an object such as {"hypothesis":"...","next_experiment":"..."}; use write for plain text')
                candidate = {**self.notes, **({key: value} if key else value)}
                if len(candidate) > 128 or len(json.dumps(candidate, ensure_ascii=False)) > 32000:
                    raise ValueError("Scratchpad capacity exceeded (128 keys / 32000 characters)")
                self.notes = candidate
            elif action == "write":
                if not key or len(json.dumps(value, ensure_ascii=False)) > 16000:
                    raise ValueError("Scratchpad requires a key and values under 16000 characters")
                candidate = {**self.notes, key: value}
                if len(candidate) > 128 or len(json.dumps(candidate, ensure_ascii=False)) > 32000:
                    raise ValueError("Scratchpad capacity exceeded (128 keys / 32000 characters)")
                self.notes[key] = value
            elif action == "delete":
                self.notes.pop(key, None)
            elif action == "clear":
                self.notes.clear()
            elif action != "read":
                raise ValueError("Unknown scratchpad action")
            return json.dumps(self.notes.get(key) if key else self.notes, ensure_ascii=False)

    def render(self):
        return self.execute("read")[:24000]

class MemoryStore:
    def __init__(self, path, workspace, embedder=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.scope = hashlib.sha256(str(Path(workspace).resolve()).encode()).hexdigest()
        self.embedder = embedder
        self.embedding_id = getattr(embedder, "identity", "custom")
        with self._db() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY, scope TEXT NOT NULL, kind TEXT NOT NULL,
                key TEXT, text TEXT NOT NULL, vector TEXT, embedding TEXT,
                created REAL NOT NULL, expires REAL)""")
            db.execute("CREATE INDEX IF NOT EXISTS memory_scope ON memories(scope,kind)")

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.path, timeout=15)
        try:
            with db:
                yield db
        finally:
            db.close()

    def save(self, text, kind="fact", key="", ttl_days=None):
        if kind not in {"fact", "episode", "preference", "document"}:
            raise ValueError("Invalid memory kind")
        if not text.strip() or len(text) > 32000:
            raise ValueError("Memory text must contain 1..32000 characters")
        if ttl_days is not None and ttl_days <= 0:
            raise ValueError("ttl_days must be positive")
        vector = self.embedder(text) if self.embedder else None
        if vector is not None and (not vector or not all(math.isfinite(v) for v in vector)):
            raise ValueError("Invalid embedding")
        scope = "user" if kind == "preference" else self.scope
        identity = uuid.uuid4().hex
        with self._db() as db:
            if key:
                db.execute("DELETE FROM memories WHERE scope=? AND kind=? AND key=?", (scope, kind, key))
            db.execute(
                "INSERT INTO memories VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    identity,
                    scope,
                    kind,
                    key,
                    text,
                    json.dumps(vector) if vector else None,
                    self.embedding_id if vector else None,
                    time.time(),
                    time.time() + ttl_days * 86400 if ttl_days else None,
                ),
            )
        return identity

    def search(self, query, limit=6, kind=None):
        if not 1 <= limit <= 50:
            raise ValueError("limit must be between 1 and 50")
        with self._db() as db:
            rows = db.execute(
                """SELECT id,kind,key,text,vector,embedding FROM memories
                WHERE scope IN (?, 'user') AND (expires IS NULL OR expires>?)
                ORDER BY created DESC LIMIT 10000""",
                (self.scope, time.time()),
            ).fetchall()
        vector = self.embedder(query) if self.embedder and query else None
        terms = set(re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", query.lower()))
        scored = []
        for identity, category, key, text, stored, embedding_id in rows:
            if kind and category != kind:
                continue
            score = 0.0
            if vector and stored and embedding_id == self.embedding_id:
                other = json.loads(stored)
                if len(vector) == len(other):
                    norm = math.sqrt(sum(x * x for x in vector) * sum(x * x for x in other))
                    score = sum(x * y for x, y in zip(vector, other)) / norm if norm else 0
            else:
                score = sum(term in text.lower() for term in terms) / max(1, len(terms))
            if score > 0 or category == "preference" or not query:
                scored.append({"id": identity, "kind": category, "key": key, "text": text, "score": score})
        preferences = [r for r in scored if r["kind"] == "preference"][: min(3, limit)]
        others = sorted((r for r in scored if r["kind"] != "preference"), key=lambda row: row["score"], reverse=True)
        return preferences + others[: limit - len(preferences)]

    def index_document(self, source, content):
        from .runtime import check_cancelled

        rows = []
        now = time.time()
        for offset in range(0, len(content), 1600):
            check_cancelled()
            text = f"{source}:{offset}\n" + content[offset : offset + 2000]
            vector = self.embedder(text) if self.embedder else None
            if vector is not None and (not vector or not all(math.isfinite(v) for v in vector)):
                raise ValueError("Invalid embedding")
            rows.append(
                (
                    uuid.uuid4().hex,
                    self.scope,
                    "document",
                    f"{source}:{offset}",
                    text,
                    json.dumps(vector) if vector else None,
                    self.embedding_id if vector else None,
                    now,
                    None,
                )
            )
        prefix = str(source) + ":"
        with self._db() as db:
            db.execute(
                "DELETE FROM memories WHERE scope=? AND kind='document' AND substr(key,1,?)=?", (self.scope, len(prefix), prefix)
            )
            db.executemany("INSERT INTO memories VALUES (?,?,?,?,?,?,?,?,?)", rows)
        return [row[0] for row in rows]

    def forget(self, identity):
        with self._db() as db:
            cursor = db.execute("DELETE FROM memories WHERE id=? AND scope IN (?, 'user')", (identity, self.scope))
            return cursor.rowcount

    def render(self, query):
        rows = self.search(query)
        return json.dumps(rows, ensure_ascii=False)[:16000] if rows else ""
