"""Local-only Ollama adapter adapted from CoreCoder corecoder/rag/providers.py.

No dependency installation, model pull, proxy, redirect or cloud embedding path.
"""

import ipaddress
import json
import math
import time
import urllib.request
from urllib.parse import urlsplit

from .runtime import check_cancelled, interruptible_call, remaining_time


def local_endpoint(value):
    if not isinstance(value, str) or len(value) > 512:
        raise ValueError("memory_embedding_url_requires_explicit_loopback_http_port")
    url = urlsplit(value)
    try:
        local = url.hostname == "localhost" or ipaddress.ip_address(url.hostname or "").is_loopback
    except ValueError:
        local = False
    if (not local or url.scheme != "http" or url.username or url.password or url.query
            or url.fragment or url.path not in {"", "/"} or not url.port):
        raise ValueError("memory_embedding_url_requires_explicit_loopback_http_port")
    return value.rstrip("/")


def validate_memory_settings(settings):
    if settings.setdefault("memory_embedding", "ollama") not in {"ollama", "off"}:
        raise ValueError("memory_embedding must be ollama or off")
    local_endpoint(settings.setdefault("memory_embedding_url", "http://127.0.0.1:11434"))
    model = settings.setdefault("memory_embedding_model", "bge-m3:latest")
    if not isinstance(model, str) or not model or len(model) > 160 or any(c.isspace() for c in model):
        raise ValueError("invalid memory_embedding_model")
    if "cloud" in model.lower():
        raise ValueError("memory_embeddings_require_local_model")
    seconds = settings.setdefault("memory_embedding_seconds", 30)
    if type(seconds) not in {int, float} or not math.isfinite(seconds) or not 1 <= seconds <= 60:
        raise ValueError("memory_embedding_seconds must be 1..60")
    if type(settings.setdefault("memory_startup_recall", True)) is not bool:
        raise ValueError("memory_startup_recall must be boolean")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("memory_embedding_redirect_rejected")


class OllamaMemoryEmbedder:
    def __init__(self, model="bge-m3:latest", base_url="http://127.0.0.1:11434", timeout=30):
        self.model, self.base_url, self.timeout = model, local_endpoint(base_url), timeout
        self.identity = "unresolved_local_embedding"

    def _request(self, path, body, expires):
        check_cancelled()
        timeout = min(expires - time.monotonic(), remaining_time() or self.timeout)
        if timeout <= 0:
            raise TimeoutError("memory_embedding_timeout")
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(self.base_url + path, data, {"Content-Type": "application/json"})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        with opener.open(request, timeout=timeout) as response:
            content = response.read(2_000_001)
        if len(content) > 2_000_000:
            raise ValueError("memory_embedding_response_too_large")
        check_cancelled()
        return json.loads(content)

    def __call__(self, text):
        check_cancelled()
        if not isinstance(text, str) or not 1 <= len(text) <= 32768:
            raise ValueError("memory_embedding_input_limit")
        timeout = min(self.timeout, remaining_time() or self.timeout)

        def embed():
            expires = time.monotonic() + timeout
            tags = self._request("/api/tags", None, expires)
            model = next((row for row in tags.get("models", [])
                          if row.get("name") in {self.model, self.model + ":latest"}), None)
            if not model or not model.get("digest"):
                raise ValueError("memory_embedding_model_not_installed")
            if model.get("remote_host") or model.get("remote_model"):
                raise ValueError("memory_embeddings_require_local_model")
            # Pin the digest and detect retagging even within one running task.
            identity = f"ollama:{self.base_url}:{model['name']}@{model['digest']}"
            result = self._request("/api/embed", {"model": self.model, "input": [text],
                                   "truncate": False, "keep_alive": "1m", "options": {"num_thread": 2}}, expires)
            vectors = result.get("embeddings", [])
            if len(vectors) != 1:
                raise ValueError("invalid_memory_embedding_count")
            vector = vectors[0]
            if not isinstance(vector, list) or not 1 <= len(vector) <= 4096 or not all(
                    type(v) in {int, float} and math.isfinite(v) for v in vector):
                raise ValueError("invalid_memory_embedding_vector")
            norm = math.sqrt(sum(v * v for v in vector))
            if not math.isfinite(norm) or norm == 0:
                raise ValueError("invalid_memory_embedding_norm")
            # Recheck after generation: a concurrently retagged alias cannot label
            # vectors as belonging to the earlier embedding space.
            after = self._request("/api/tags", None, expires)
            if not any(row.get("name") == model["name"] and row.get("digest") == model["digest"]
                       for row in after.get("models", [])):
                raise ValueError("memory_embedding_model_changed")
            check_cancelled()
            return identity, [v / norm for v in vector]

        identity, vector = interruptible_call(embed, timeout=timeout)
        check_cancelled()
        self.identity = identity
        return vector
