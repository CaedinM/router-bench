#!/usr/bin/env python3
"""Upload all question groups with fixed development/test memberships."""
import gzip
import json
import os
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('PHOENIX_WORKING_DIR', str(ROOT / 'phoenix/storage'))
os.environ.setdefault('PHOENIX_BASE_URL', 'http://127.0.0.1:6006')
from phoenix.client import Client
from split_selection import load_development

NAME = 'router-bench-intern-flash-gpt5-full-v1'
PREFIX = {'intern-s1': 'intern', 'gemini-2.5-flash': 'flash', 'gpt-5': 'gpt5'}
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
    example = {
        'input': {'prompt': baseline['prompt']},
        'output': {f'{PREFIX[model]}_{field}': value for model, record in records.items() for field, value in record.items()},
        'metadata': {'dataset_id': key[0], 'split': key[1], 'record_index': key[2],
                     'prompt_alignment_exact': len({r['prompt'] for r in records.values()}) == 1,
                     'origin_query_alignment_exact': len({r['origin_query'] for r in records.values()}) == 1},
        'splits': ['development' if key in development else 'test'],
    }
    examples.append(example)
client = Client()
if any(dataset['name'] == NAME for dataset in client.datasets.list()):
    raise ValueError(f'Dataset already exists: {NAME}; refusing to append duplicate examples')
for start in range(0, len(examples), 2000):
    batch = examples[start:start + 2000]
    if not start:
        result = client.datasets.create_dataset(name=NAME, examples=batch, timeout=600,
            dataset_description='Full three-model benchmark; Intern-S1 / Gemini 2.5 Flash / GPT-5 Medium; prior 30 development memberships retained, remaining questions in test.')
    else:
        result = client.datasets.add_examples_to_dataset(dataset=NAME, examples=batch, timeout=600)
    print(f'Uploaded {min(start + len(batch), len(examples))}/{len(examples)} examples', flush=True)
print(result, flush=True)
