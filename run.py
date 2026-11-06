"""RepoAgent portable entrypoint; external dependencies stay in your virtual environment."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'runtime/app'))
from repoagent.cli import main

if __name__ == '__main__':
    raise SystemExit(main())
