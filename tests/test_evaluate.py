import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from evaluators.downstream_metrics import evaluate_experiment, published_metrics


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.reference = {
            f'{prefix}_{field}': value
            for prefix, values in {
                'intern': (0.5, 10, 20, 0.001),
                'flash': (1, 30, 40, 0.02),
                'gpt5': (0, 50, 60, 0.03),
            }.items()
            for field, value in zip(('score', 'prompt_tokens', 'completion_tokens', 'cost'), values)
        }

    def test_copies_each_selected_models_values(self):
        for label, model, values in (
            ('Candidate A', 'intern-s1', (0.5, 10, 20, 0.001)),
            ('Candidate B', 'gemini-2.5-flash', (1, 30, 40, 0.02)),
            ('Candidate C', 'gpt-5', (0, 50, 60, 0.03)),
        ):
            with self.subTest(label=label):
                results = published_metrics({'decision': label, 'selected_model': model}, self.reference)
                self.assertEqual([result['score'] for result in results], list(values))
                self.assertEqual([result['name'] for result in results], [
                    'downstream_score', 'downstream_prompt_tokens',
                    'downstream_completion_tokens', 'downstream_cost'])
                self.assertTrue(all(not result['metadata']['invalid_usage'] for result in results))

    def test_flags_invalid_usage_for_all_candidate_choices(self):
        self.reference['gpt5_completion_tokens'] = -1
        results = published_metrics({'decision': 'Candidate A', 'selected_model': 'intern-s1'}, self.reference)
        self.assertTrue(all(result['metadata']['invalid_usage'] for result in results))
        self.assertEqual(results[2]['score'], 20)

    def test_rejects_missing_values_and_mismatched_decisions(self):
        with self.assertRaises(ValueError):
            published_metrics({'decision': 'Candidate A', 'selected_model': 'gpt-5'}, self.reference)
        del self.reference['intern_cost']
        with self.assertRaises(ValueError):
            published_metrics({'decision': 'Candidate A', 'selected_model': 'intern-s1'}, self.reference)

    def test_evaluates_existing_experiment(self):
        experiments = Mock()
        client = SimpleNamespace(experiments=experiments)
        evaluate_experiment(client, 'experiment-id')
        experiments.get_experiment.assert_called_once_with(experiment_id='experiment-id')
        experiments.evaluate_experiment.assert_called_once_with(
            experiment=experiments.get_experiment.return_value, evaluators=[published_metrics],
            print_summary=False, retries=0)


if __name__ == '__main__':
    unittest.main()
