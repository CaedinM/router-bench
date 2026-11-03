"""Opt-in live test: Phoenix development split -> router API -> persisted decisions."""
import json
import math
import os
from pathlib import Path
import unittest
from urllib.parse import quote
from uuid import uuid4

from routing.candidates import CANDIDATE_MODELS
from routing.pipeline import (
    ROUTING_CONCURRENCY,
    load_dataset,
    load_environment,
    make_phoenix_client,
    run_pipeline_async,
    select_router_config,
)


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.environ.get('RUN_ROUTING_E2E') == '1',
                     'Set RUN_ROUTING_E2E=1 to create a live, paid 30-question experiment')
class RoutingEndToEndTests(unittest.IsolatedAsyncioTestCase):
    async def test_development_decisions_are_saved_in_phoenix_and_jsonl(self):
        load_environment()
        config_path = Path(os.environ.get('ROUTING_E2E_CONFIG', ROOT / 'routing/config.json'))
        config = json.loads(config_path.read_text())
        # Never inherit the full test split from the regular run configuration.
        config['split'] = 'development'
        router_name = os.environ.get('ROUTING_E2E_ROUTER', 'jev')
        select_router_config(config, router_name)
        output = ROOT / 'routing/results' / f'e2e-{uuid4().hex}.jsonl'
        print(f'\nLive routing test: router={router_name}, log={output}', flush=True)

        client = make_phoenix_client(config, asynchronous=True)
        try:
            dataset = await load_dataset(client, config)
            # Fail before calling the provider if Phoenix has the wrong split.
            self.assertEqual(len(dataset), 30, 'Expected exactly 30 development questions')
            example_ids = {example['id'] for example in dataset}
            self.assertEqual(len(example_ids), 30)
            config['dataset_version'] = dataset.version_id

            experiment = (await run_pipeline_async(
                config, output, router_name=router_name, client=client))[router_name]
            experiment_id = experiment['experiment_id']
            print(f'Live routing experiment: {experiment_id}', flush=True)

            # Read back from Phoenix to verify persistence, not just returned task outputs.
            saved = await client.experiments.get(experiment_id=experiment_id)
            self.assertEqual(saved['dataset_id'], dataset.id)
            self.assertEqual(saved['dataset_version_id'], dataset.version_id)
            self.assertEqual(saved['metadata']['split'], 'development')
            self.assertEqual(saved['metadata']['concurrency'], ROUTING_CONCURRENCY)
            runs, cursor = [], None
            while True:
                params = {'limit': 100}
                if cursor:
                    params['cursor'] = cursor
                response = await client._client.get(
                    f'v1/experiments/{quote(experiment_id, safe="")}/runs',
                    params=params, timeout=600)
                response.raise_for_status()
                page = response.json()
                runs.extend(page['data'])
                cursor = page.get('next_cursor')
                if not cursor:
                    break

            self.assertEqual(len(runs), 30)
            self.assertEqual({run['dataset_example_id'] for run in runs}, example_ids)
            records = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual({record['example_id'] for record in records}, example_ids)
            # Phoenix retries can leave failed attempts in the append-only log.
            successful = {record['example_id']: record for record in records if record['status'] == 'ok'}
            self.assertEqual(set(successful), example_ids)
            for record in records:
                self.assertEqual(record['split'], 'development')
                self.assertEqual(record['dataset_version'], dataset.version_id)
                self.assertEqual(record['router'], router_name)
                self.assertEqual(record['concurrency'], ROUTING_CONCURRENCY)
            for run in runs:
                with self.subTest(example_id=run['dataset_example_id']):
                    self.assertFalse(run.get('error'), f'Phoenix task failed: {run.get("error")}')
                    result = run['output']
                    self.assertIn(result['decision'], CANDIDATE_MODELS)
                    self.assertEqual(result['selected_model'], CANDIDATE_MODELS[result['decision']])
                    self.assertTrue(math.isfinite(result['latency_seconds']))
                    self.assertGreaterEqual(result['latency_seconds'], 0)
                    record = successful[run['dataset_example_id']]
                    for key in ('decision', 'selected_model', 'usage', 'latency_seconds'):
                        self.assertEqual(result[key], record[key])
        finally:
            await client._client.aclose()


if __name__ == '__main__':
    unittest.main()
