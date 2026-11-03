"""Run a Haiku routing experiment using the existing Phoenix pipeline."""
import argparse
import json
from pathlib import Path
from uuid import uuid4

from routing.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path,
                        default=Path(__file__).with_name('config.json'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = args.output or Path('routing/llm_routing/results') / f'{uuid4().hex}.jsonl'
    run_pipeline(config, output, router_name='haiku')
    print(f'Decisions saved to {output.resolve()}')


if __name__ == '__main__':
    main()
