"""Small pi-inspired event surface; atomic local records contain metadata only."""

import copy
import threading
import time
from pathlib import Path

from .task_state import atomic_json


class Events:
    def __init__(self, directory):
        self.directory = Path(directory) / "events"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.sequence = max((int(p.stem) for p in self.directory.glob("*.json")), default=0)
        self.listeners = []

    def subscribe(self, callback):
        self.listeners.append(callback)

    def emit(self, kind, **metadata):
        with self.lock:
            self.sequence += 1
            event = {"id": self.sequence, "type": kind, "time": time.time(), **metadata}
            atomic_json(self.directory / f"{self.sequence:06d}.json", event)
            for callback in self.listeners:
                callback(copy.deepcopy(event))
            return event["id"]
