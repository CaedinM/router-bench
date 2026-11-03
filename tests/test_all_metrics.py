import asyncio
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call, patch

from evaluators.all_metrics import evaluate_experiment, run_evaluation


class AllMetricsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.reference = {
            f'{prefix}_{field}': value
            for prefix, values in {'intern': (0.5, 10, 20, 0.001),
                                   'flash': (1, 30, 40, 0.02),
                                   'gpt5': (0, 50, 60, 0.03)}.items()
            for field, value in zip(('score', 'prompt_tokens', 'completion_tokens', 'cost'), values)
        }
        self.output = {'decision': 'Candidate A', 'selected_model': 'intern-s1',
                       'latency_seconds': 1, 'usage': {'input_tokens': 1000, 'output_tokens': 42}}
        self.experiment = {
            'dataset_id': 'saved-dataset', 'dataset_version_id': 'saved-version',
            'experiment_metadata': {'router_config': {'provider': 'jev', 'model': 'jev-1.13.0'}},
            'task_runs': [{'id': 'run-1', 'dataset_example_id': 'example-1', 'output': self.output}],
        }
        self.experiments = SimpleNamespace(
            get=AsyncMock(return_value={
                'dataset_id': 'saved-dataset', 'dataset_version_id': 'saved-version',
                'metadata': self.experiment['experiment_metadata']}), log_evaluation=AsyncMock())
        self.datasets = SimpleNamespace(get_dataset=AsyncMock(return_value=[
            {'id': 'example-1', 'output': self.reference}]))
        self.response = Mock()
        self.response.json.side_effect = lambda: {'data': self.experiment['task_runs']}
        self.client = SimpleNamespace(experiments=self.experiments, datasets=self.datasets,
                                      _client=SimpleNamespace(aclose=AsyncMock(),
                                          get=AsyncMock(return_value=self.response)))

    def submissions(self):
        return {call.kwargs['name']: call.kwargs for call in self.experiments.log_evaluation.await_args_list}

    async def test_fetches_pinned_references_once_and_preserves_all_metrics(self):
        self.reference['gpt5_completion_tokens'] = -1
        original = copy.deepcopy(self.reference)
        summary = await evaluate_experiment(self.client, 'experiment-id')
        self.experiments.get.assert_awaited_once_with(experiment_id='experiment-id')
        self.client._client.get.assert_awaited_once_with(
            'v1/experiments/experiment-id/runs', params={'limit': 1000}, timeout=600)
        self.datasets.get_dataset.assert_awaited_once_with(
            dataset='saved-dataset', version_id='saved-version', timeout=600)
        results = self.submissions()
        self.assertEqual({name: call['score'] for name, call in results.items()}, {
            'downstream_score': 0.5, 'downstream_prompt_tokens': 10,
            'downstream_completion_tokens': 20, 'downstream_cost': 0.001,
            'router_latency_seconds': 1, 'router_input_tokens': 1000,
            'router_output_tokens': 42, 'routing_cost': 0.000042,
            'total_system_cost': 0.001042, 'gap': 0.5,
        })
        self.assertEqual(results['downstream_cost']['metadata'], {
            'selected_model': 'intern-s1', 'invalid_usage': True})
        self.assertTrue(results['total_system_cost']['metadata']['invalid_usage'])
        self.assertEqual(results['routing_cost']['metadata']['model'], 'jev-1.13.0')
        for call in results.values():
            self.assertEqual(call['experiment_run_id'], 'run-1')
            self.assertEqual(call['timeout'], 600)
            self.assertLessEqual(set(call), {'experiment_run_id', 'name', 'score', 'metadata', 'timeout'})
        self.assertEqual(self.reference, original)
        self.assertEqual(summary, {'evaluated_runs': 1, 'skipped_failed_runs': 0,
                                   'submitted_evaluations': 10, 'evaluation_errors': 0})

    async def test_metric_failures_are_logged_without_blocking_other_metrics(self):
        self.output['usage'] = None
        summary = await evaluate_experiment(self.client, 'experiment-id')
        results = self.submissions()
        self.assertEqual(results['downstream_cost']['score'], 0.001)
        self.assertEqual(results['router_latency_seconds']['score'], 1)
        self.assertEqual(results['gap']['score'], 0.5)
        for name in ('routing_cost_metric', 'total_system_cost'):
            self.assertIn('Router usage is missing', results[name]['error'])
            self.assertNotIn('score', results[name])
        self.assertNotIn('router_input_tokens', results)
        self.assertEqual(summary['evaluation_errors'], 2)
        self.assertEqual(summary['submitted_evaluations'], 8)

    async def test_failed_tasks_are_skipped_and_repeated_runs_keep_their_ids(self):
        self.experiment['task_runs'].extend([
            {'id': 'failed-run', 'dataset_example_id': 'missing-example', 'error': 'refusal'},
            {'id': 'run-2', 'dataset_example_id': 'example-1', 'output': self.output},
        ])
        summary = await evaluate_experiment(self.client, 'experiment-id')
        calls = self.experiments.log_evaluation.await_args_list
        self.assertEqual({call.kwargs['experiment_run_id'] for call in calls}, {'run-1', 'run-2'})
        self.assertEqual(summary['submitted_evaluations'], 20)
        self.assertEqual(summary['skipped_failed_runs'], 1)

    async def test_missing_reference_does_not_block_router_metrics_or_routing_cost(self):
        self.datasets.get_dataset.return_value = []
        summary = await evaluate_experiment(self.client, 'experiment-id')
        results = self.submissions()
        self.assertEqual(results['router_input_tokens']['score'], 1000)
        self.assertEqual(results['routing_cost']['score'], 0.000042)
        self.assertIn('error', results['published_metrics'])
        self.assertIn('error', results['total_system_cost'])
        self.assertIn('error', results['gap'])
        self.assertEqual(summary['evaluation_errors'], 3)

    async def test_submissions_overlap_within_the_concurrency_limit(self):
        saturated, release = asyncio.Event(), asyncio.Event()
        active = peak = 0

        async def log(**kwargs):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            if active == 2:
                saturated.set()
            try:
                await release.wait()
            finally:
                active -= 1

        self.experiments.log_evaluation.side_effect = log
        task = asyncio.create_task(evaluate_experiment(self.client, 'experiment-id', concurrency=2))
        try:
            await asyncio.wait_for(saturated.wait(), timeout=1)
            self.assertEqual(active, 2)
        finally:
            release.set()
            await asyncio.wait_for(task, timeout=1)
        self.assertEqual(peak, 2)
        self.assertEqual(active, 0)

    async def test_submission_failure_propagates_and_cancels_outstanding_writes(self):
        cancelled = asyncio.Event()

        async def log(**kwargs):
            if kwargs['name'] == 'downstream_score':
                raise RuntimeError('Phoenix unavailable')
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        self.experiments.log_evaluation.side_effect = log
        with self.assertRaisesRegex(RuntimeError, 'Phoenix unavailable'):
            await asyncio.wait_for(evaluate_experiment(self.client, 'experiment-id', concurrency=2), timeout=1)
        self.assertTrue(cancelled.is_set())
        self.assertEqual(self.experiments.log_evaluation.await_count, 2)

    async def test_run_pagination_keeps_all_pages_without_a_full_export(self):
        second = {'id': 'run-2', 'dataset_example_id': 'example-1', 'output': self.output}
        first_page, second_page = Mock(), Mock()
        first_page.json.return_value = {'data': self.experiment['task_runs'], 'next_cursor': 'page-2'}
        second_page.json.return_value = {'data': [second], 'next_cursor': None}
        self.client._client.get.side_effect = [first_page, second_page]
        summary = await evaluate_experiment(self.client, 'experiment-id')
        self.assertEqual(summary['evaluated_runs'], 2)
        self.assertEqual(summary['submitted_evaluations'], 20)
        self.assertEqual(self.client._client.get.await_args_list, [
            call('v1/experiments/experiment-id/runs', params={'limit': 1000}, timeout=600),
            call('v1/experiments/experiment-id/runs', params={'limit': 1000, 'cursor': 'page-2'}, timeout=600),
        ])

    async def test_invalid_concurrency_fails_before_fetching(self):
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                await evaluate_experiment(self.client, 'experiment-id', concurrency=value)
        self.experiments.get.assert_not_awaited()

    async def test_cli_client_is_closed_after_success_or_failure(self):
        for error in (None, RuntimeError('fetch failed')):
            with self.subTest(error=error):
                self.client._client.aclose.reset_mock()
                self.experiments.get.side_effect = error
                with patch('evaluators.all_metrics.make_phoenix_client', return_value=self.client) as factory:
                    if error:
                        with self.assertRaisesRegex(RuntimeError, 'fetch failed'):
                            await run_evaluation({}, 'experiment-id', 2)
                    else:
                        await run_evaluation({}, 'experiment-id', 2)
                    factory.assert_called_once_with({}, asynchronous=True)
                    self.client._client.aclose.assert_awaited_once()


if __name__ == '__main__':
    unittest.main()
