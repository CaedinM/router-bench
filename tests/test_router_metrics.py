import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from evaluators.router_metrics import evaluate_experiment, router_metrics


class RouterMetricsTests(unittest.TestCase):
    def test_copies_available_metrics(self):
        result = router_metrics({'latency_seconds': 1.25, 'usage': {'input_tokens': 100, 'output_tokens': 0}})
        self.assertEqual(result, [
            {'name': 'router_latency_seconds', 'score': 1.25},
            {'name': 'router_input_tokens', 'score': 100},
            {'name': 'router_output_tokens', 'score': 0},
        ])

    def test_missing_usage_stays_missing(self):
        self.assertEqual(router_metrics({'latency_seconds': 2, 'usage': None}),
                         [{'name': 'router_latency_seconds', 'score': 2}])
        self.assertEqual(router_metrics({'latency_seconds': 2, 'usage': {'input_tokens': 3}}), [
            {'name': 'router_latency_seconds', 'score': 2},
            {'name': 'router_input_tokens', 'score': 3},
        ])

    def test_rejects_invalid_values(self):
        for output in ({'latency_seconds': -1}, {'latency_seconds': 1, 'usage': {'input_tokens': -2}},
                       {'latency_seconds': 1, 'usage': {'output_tokens': True}}):
            with self.subTest(output=output), self.assertRaises(ValueError):
                router_metrics(output)

    def test_evaluates_existing_experiment(self):
        experiments = Mock()
        evaluate_experiment(SimpleNamespace(experiments=experiments), 'experiment-id')
        experiments.get_experiment.assert_called_once_with(experiment_id='experiment-id')
        experiments.evaluate_experiment.assert_called_once_with(
            experiment=experiments.get_experiment.return_value, evaluators=[router_metrics],
            print_summary=False, retries=0)


if __name__ == '__main__':
    unittest.main()
