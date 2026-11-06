import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor"))
os.environ["PYTHON_DOTENV_DISABLED"] = "1"
os.environ["MSWEA_GLOBAL_CONFIG_DIR"] = str(Path(__file__).resolve().parents[2] / ".tmp/mini-config")
os.environ["MSWEA_SILENT_STARTUP"] = "1"
