"""Offline file-localization evaluation; labels never enter the agent prompt."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'runtime/app'))
from corecoder_mini.task_state import atomic_json
from repoagent.localization import CodeIndex
from repoagent.metrics import localization


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('-k', type=int, default=5)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('new_evaluation_output_required')
    cases = [json.loads(line) for line in args.cases.read_text(encoding='utf-8').splitlines() if line.strip()]
    args.output.mkdir(parents=True)
    index = CodeIndex(args.workspace, args.output / 'index.json')
    result = localization(index, cases, args.k)
    atomic_json(args.output / 'result.json', result)
    print(json.dumps(result['aggregation'], ensure_ascii=False))
if __name__ == '__main__':
    main()
