"""Select our pinned vendored package without reading the user's global config."""

import logging
import os
import sys
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
os.environ["MSWEA_GLOBAL_CONFIG_DIR"] = str(APP.parent / ".tmp" / "mini-config")
os.environ["MSWEA_SILENT_STARTUP"] = "1"
sys.path.insert(0, str(APP / "vendor"))

from minisweagent.agents.default import DefaultAgent
from minisweagent.exceptions import FormatError, Submitted
from minisweagent.models.utils.actions_toolcall import (
    BASH_TOOL,
    format_toolcall_observation_messages,
    parse_toolcall_actions,
)

logging.getLogger("agent").setLevel(logging.WARNING)

__all__ = [
    "BASH_TOOL",
    "DefaultAgent",
    "FormatError",
    "Submitted",
    "format_toolcall_observation_messages",
    "parse_toolcall_actions",
]
