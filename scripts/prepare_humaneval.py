"""Explicitly provision pinned official data; never run model inference."""
import argparse
import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / 'third_party/human_eval'
DATA = 'data/HumanEval.jsonl.gz'
MAX_BYTES = 16 * 1024 * 1024


def install(raw, official=OFFICIAL):
    official = Path(official)
    manifest = json.loads((official / 'UPSTREAM.json').read_text(encoding='utf-8'))
    expected = manifest['files'][DATA]
    if len(raw) > MAX_BYTES or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('official_humaneval_data_hash_mismatch')
    target = official / DATA
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() == expected:
            return target
        raise ValueError('existing_humaneval_data_differs; preserve and inspect it manually')
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.humaneval-', dir=target.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw)
        os.replace(name, target)
    finally:
        Path(name).unlink(missing_ok=True)
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--data-file', type=Path)
    group.add_argument('--download', action='store_true', help='Explicitly permit a pinned upstream download')
    args = parser.parse_args(argv)
    if args.data_file:
        with args.data_file.open('rb') as stream:
            raw = stream.read(MAX_BYTES + 1)
    else:
        manifest = json.loads((OFFICIAL / 'UPSTREAM.json').read_text(encoding='utf-8'))
        url = 'https://raw.githubusercontent.com/' + manifest['repo'] + '/' + manifest['commit'] + '/' + DATA
        with urllib.request.urlopen(url, timeout=60) as response:
            raw = response.read(MAX_BYTES + 1)
    target = install(raw)
    print('Pinned official HumanEval data verified: ' + str(target))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
