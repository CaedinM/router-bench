#!/usr/bin/env python3
"""Upload all question groups with fixed development/test memberships."""
import gzip
import json
import os
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
from routing.pipeline import load_environment, make_phoenix_client
from routing.split_selection import load_development
from scripts.load_benchmarks import BENCHMARKS

NAME = 'router-bench-intern-flash-gpt5-full-v1'
PREFIX = {'intern-s1': 'intern', 'gemini-2.5-flash': 'flash', 'gpt-5': 'gpt5'}


def upload_dataset(*, reuse_existing=True):
    """Create fixed partitions and benchmark splits, including on existing datasets."""
    load_environment()
    os.environ.setdefault('PHOENIX_BASE_URL', 'http://127.0.0.1:6006')
    client = make_phoenix_client({})
    try:
        if any(dataset['name'] == NAME for dataset in client.datasets.list(timeout=600)):
            if not reuse_existing:
                raise ValueError(f'Dataset already exists: {NAME}; refusing to append duplicate examples')
            endpoint = f'v1/datasets/{quote(NAME, safe="")}/splits'
            existing_splits = set()
            cursor = None
            while True:
                response = client._client.get(endpoint, params={'limit': 100, 'cursor': cursor}, timeout=600)
                response.raise_for_status()
                page = response.json()
                existing_splits.update(item['name'] for item in page['data'])
                cursor = page.get('next_cursor')
                if not cursor:
                    break
            for partition in ('development', 'test'):
                missing = {f'{partition}/{benchmark}' for benchmark in BENCHMARKS} - existing_splits
                if not missing:
                    continue
                dataset = client.datasets.get_dataset(dataset=NAME, splits=[partition], timeout=600)
                memberships = defaultdict(list)
                for example in dataset:
                    memberships[f'{partition}/{example["metadata"]["dataset_id"]}'].append(example['id'])
                for name in sorted(missing):
                    if not memberships[name]:
                        raise ValueError(f'No examples for benchmark split: {name}')
                    response = client._client.post(
                        endpoint, json={'name': name, 'example_ids': memberships[name]}, timeout=600)
                    response.raise_for_status()
                    print(f'Created benchmark split: {name} ({len(memberships[name])} examples)')
            print(f'Reusing dataset: {NAME}')
            return NAME
        groups = defaultdict(dict)
        with gzip.open(ROOT / 'data/intern_flash_gpt5/records.jsonl.gz', 'rt', encoding='utf-8') as f:
            for line in f:
                row = json.loads(line)
                key = (row['dataset_id'], row['split'], row['record_index'])
                if row['model_name'] not in PREFIX or row['model_name'] in groups[key]:
                    raise ValueError(f'Unexpected or duplicate model at {key}')
                groups[key][row['model_name']] = row
        if len(groups) != 12446:
            raise ValueError('Unexpected question count')
        # Keep the prior split fixed instead of selecting questions using new-pool scores.
        development = load_development()
        if not development <= groups.keys():
            raise ValueError('Development questions missing from active dataset')
        examples = []
        for key, records in sorted(groups.items()):
            if set(records) != set(PREFIX):
                raise ValueError(f'Missing model at {key}')
            baseline = records['gemini-2.5-flash']
            partition = 'development' if key in development else 'test'
            example = {
                'input': {'prompt': baseline['prompt']},
                'output': {f'{PREFIX[model]}_{field}': value for model, record in records.items() for field, value in record.items()},
                'metadata': {'dataset_id': key[0], 'split': key[1], 'record_index': key[2],
                             'prompt_alignment_exact': len({r['prompt'] for r in records.values()}) == 1,
                             'origin_query_alignment_exact': len({r['origin_query'] for r in records.values()}) == 1},
                'splits': [partition, f'{partition}/{key[0]}'],
            }
            examples.append(example)
        for start in range(0, len(examples), 2000):
            batch = examples[start:start + 2000]
            if not start:
                client.datasets.create_dataset(name=NAME, examples=batch, timeout=600,
                    dataset_description='Full three-model benchmark; Intern-S1 / Gemini 2.5 Flash / GPT-5 Medium; prior 30 development memberships retained, remaining questions in test.')
            else:
                client.datasets.add_examples_to_dataset(dataset=NAME, examples=batch, timeout=600)
            print(f'Uploaded {min(start + len(batch), len(examples))}/{len(examples)} examples', flush=True)
        return NAME
    finally:
        client._client.close()


if __name__ == '__main__':
    print(upload_dataset(reuse_existing=False), flush=True)
