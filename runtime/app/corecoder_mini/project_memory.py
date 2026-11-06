"""Hybrid retrieval over the existing scoped memory schema.

Reuses MemoryStore persistence and CoreCoder RAG's SQLite FTS5/BM25 + weighted
reciprocal rank fusion pattern. Exact cosine avoids a mandatory vector library.
"""

import json
import math
import re
import sqlite3
import time
from contextlib import closing

from .legacy_memory import MemoryStore
from .runtime import Cancelled, check_cancelled


def terms(text):
    # Adapted from corecoder/rag/parsing.py:terms (identifiers + Chinese bigrams).
    separated = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    tokens = re.findall(r"[a-zA-Z0-9_]+", separated.lower())
    tokens += [part for token in tokens.copy() for part in token.split("_") if part and part != token]
    tokens += re.findall(r"[a-zA-Z][a-zA-Z0-9_]*", text.lower())
    for run in re.findall(r"[\u3400-\u9fff]+", text):
        tokens.extend(run)
        tokens.extend(run[i:i + 2] for i in range(len(run) - 1))
    return tokens


class HybridMemoryStore(MemoryStore):
    window = 2000

    def __init__(self, path, workspace, embedder=None):
        super().__init__(path, workspace)
        self.provider, self.embedding_error, self.retry_at = embedder, None, 0.0
        self.embedder = self._embed if embedder else None
        self.embedding_id = getattr(embedder, "identity", "off") if embedder else "off"
        self.last_trace = {}

    def _embed(self, text):
        check_cancelled()
        if time.monotonic() < self.retry_at:
            return None
        try:
            vector = self.provider(text)
            if not isinstance(vector, (list, tuple)) or not 1 <= len(vector) <= 4096 or not all(
                    type(v) in {int, float} and math.isfinite(v) for v in vector):
                raise ValueError("invalid_memory_embedding")
            norm = math.sqrt(sum(v * v for v in vector))
            if not math.isfinite(norm) or norm == 0:
                raise ValueError("invalid_memory_embedding_norm")
            identity = getattr(self.provider, "identity", None)
            if not isinstance(identity, str) or not identity or identity == "unresolved_local_embedding":
                raise ValueError("memory_embedding_identity_required")
            check_cancelled()
            self.embedding_id, self.embedding_error = identity, None
            return [v / norm for v in vector]
        except Cancelled:
            raise
        except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
            # Never expose request text, server bodies or private paths in errors.
            self.embedding_error = type(exc).__name__
            self.retry_at = time.monotonic() + 30
            return None

    def save(self, *args, **kwargs):
        check_cancelled()
        result = super().save(*args, **kwargs)
        check_cancelled()
        return result

    def _rows(self, kind=None):
        with self._db() as db:
            rows = db.execute("""SELECT id,kind,key,text,vector,embedding FROM memories
                WHERE scope=? AND (expires IS NULL OR expires>?) AND (? IS NULL OR kind=?)
                ORDER BY created DESC,id LIMIT ?""", (self.scope, time.time(), kind, kind, self.window + 1)).fetchall()
        return rows[:self.window], len(rows) > self.window

    def status(self):
        rows, truncated = self._rows()
        return {"retrieval": "weighted_rrf:vector+bm25+keyword", "project_records": len(rows),
                "window_limit": self.window, "window_truncated": truncated,
                "vector_records": sum(bool(row[4]) for row in rows),
                "embedding": self.embedding_id if self.provider else "off",
                "embedding_error": self.embedding_error, "retry_backoff": time.monotonic() < self.retry_at,
                "last_search": self.last_trace}

    def _stored_vector(self, row, dimension):
        if not row[4] or row[5] != self.embedding_id:
            return None
        try:
            other = json.loads(row[4])
            if not isinstance(other, list) or len(other) != dimension or not all(
                    type(v) in {int, float} and math.isfinite(v) for v in other):
                return None
            norm = math.sqrt(sum(v * v for v in other))
            if not math.isfinite(norm) or norm == 0:
                return None
            return [v / norm for v in other]
        except (ValueError, TypeError, OverflowError):
            return None

    def search(self, query, limit=6, kind=None):
        check_cancelled()
        if not isinstance(query, str) or len(query) > 1000 or not 1 <= limit <= 50:
            raise ValueError("memory_search_limits")
        started = time.monotonic()
        rows, truncated = self._rows(kind)
        self.last_trace = {"records_scanned": len(rows), "window_truncated": truncated,
                           "vector": 0, "bm25": 0, "keyword": 0, "incompatible_vectors": 0}
        if not query.strip() or not rows:
            return [{"id": row[0], "kind": row[1], "key": row[2], "text": row[3], "score": 0,
                     "channels": {}, "embedding_error": self.embedding_error} for row in rows[:limit]] if not query.strip() else []
        wanted = set(terms(query))
        vector = self._embed(query) if self.provider else None
        dense, keyword = {}, {}
        for index, row in enumerate(rows):
            check_cancelled()
            words = set(terms((row[2] or "") + " " + row[3]))
            overlap = len(wanted & words) / max(1, len(wanted))
            if overlap:
                keyword[index] = overlap + (0.5 if query.casefold() in ((row[2] or "") + " " + row[3]).casefold() else 0)
            if vector and row[4]:
                other = self._stored_vector(row, len(vector))
                if other:
                    cosine = sum(a * b for a, b in zip(vector, other))
                    if math.isfinite(cosine) and cosine > 0:
                        dense[index] = cosine
                else:
                    self.last_trace["incompatible_vectors"] += 1
        sparse = {}
        # A temporary scoped FTS corpus prevents other projects changing BM25's
        # document frequencies and makes old SQLite notes searchable immediately.
        if wanted:
            try:
                with closing(sqlite3.connect(":memory:")) as db:
                    db.execute("CREATE VIRTUAL TABLE candidates USING fts5(tokens)")
                    db.executemany("INSERT INTO candidates(rowid,tokens) VALUES(?,?)",
                                   [(i + 1, " ".join(terms((row[2] or "") + " " + row[3]))) for i, row in enumerate(rows)])
                    expression = " OR ".join('"' + word.replace('"', '""') + '"' for word in sorted(wanted)[:128])
                    for identity, score in db.execute("SELECT rowid,bm25(candidates) FROM candidates WHERE candidates MATCH ?",
                                                       (expression,)):
                        sparse[identity - 1] = -score
            except sqlite3.OperationalError as exc:
                # FTS5 is optional in some Python SQLite builds.
                self.last_trace["bm25_error"] = type(exc).__name__
        check_cancelled()
        fused = {}
        channels = {"vector": dense, "bm25": sparse, "keyword": keyword}
        for name, weight in [("vector", .5), ("bm25", .35), ("keyword", .15)]:
            self.last_trace[name] = len(channels[name])
            for rank, index in enumerate(sorted(channels[name], key=lambda i: (-channels[name][i], i))[:64], 1):
                fused[index] = fused.get(index, 0) + weight / (60 + rank)
        self.last_trace.update({"returned": min(limit, len(fused)), "embedding_error": self.embedding_error,
                                "seconds": round(time.monotonic() - started, 4)})
        return [{"id": rows[i][0], "kind": rows[i][1], "key": rows[i][2], "text": rows[i][3],
                 "score": fused[i], "channels": {name: values[i] for name, values in channels.items() if i in values},
                 "embedding_error": self.embedding_error}
                for i in sorted(fused, key=lambda i: (-fused[i], i))[:limit]]

    def reindex(self, limit=8):
        """Explicit bounded migration, CAS against concurrent text replacement."""
        check_cancelled()
        if type(limit) is not int or not 1 <= limit <= 16:
            raise ValueError("memory_reindex_limit_1_to_16")
        if not self.provider:
            return {"updated": 0, "embedding_error": "embedding_off"}
        rows, truncated = self._rows()
        # Resolve the current digest once; cached vectors must never resolve it.
        vector = self._embed("embedding space identity check") if rows else None
        if rows and vector is None:
            return {"updated": 0, "embedding_error": self.embedding_error}
        pending = [row for row in rows if self._stored_vector(row, len(vector)) is None]
        updated = 0
        for row in pending[:limit]:
            vector = self._embed(row[3])
            if vector is None:
                break
            check_cancelled()
            with self._db() as db:
                updated += db.execute("UPDATE memories SET vector=?,embedding=? WHERE id=? AND scope=? AND text=?",
                                      (json.dumps(vector), self.embedding_id, row[0], self.scope, row[3])).rowcount
        return {"updated": updated, "remaining_in_window": len(pending) - updated,
                "window_truncated": truncated, "embedding_error": self.embedding_error}

    def forget(self, identity):
        check_cancelled()
        with self._db() as db:
            return db.execute("DELETE FROM memories WHERE id=? AND scope=?", (identity, self.scope)).rowcount
