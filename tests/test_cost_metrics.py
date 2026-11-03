import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from evaluators.cost_metrics import evaluate_experiment, gap, make_cost_evaluators, routing_cost


class CostMetricsTests(unittest.TestCase):
    def setUp(self):
        self.jev = {'provider': 'jev', 'model': 'jev-1.13.0'}
        self.openai = {'provider': 'openai_decisions', 'model': 'gpt-6-luna'}
        self.output = {'decision': 'Candidate A', 'selected_model': 'intern-s1',
                       'usage': {'input_tokens': 1000, 'output_tokens': 42}}
        self.reference = {
            f'{prefix}_{field}': value
            for prefix, values in {'intern': (0.5, 10, 20, 0.001),
                                   'flash': (1, 30, 40, 0.02),
                                   'gpt5': (0, 50, 60, 0.03)}.items()
            for field, value in zip(('score', 'prompt_tokens', 'completion_tokens', 'cost'), values)
        }

    def test_jev_input_only_pricing(self):
        self.assertAlmostEqual(routing_cost(self.output, self.jev)['score'], 0.000042)
        self.output['usage'].pop('output_tokens')
        self.assertAlmostEqual(routing_cost(self.output, self.jev)['score'], 0.000042)

    def test_openai_cache_reads_and_writes_are_not_double_counted(self):
        self.output['usage']['input_tokens_details'] = {'cached_tokens': 400, 'cache_write_tokens': 200}
        self.assertAlmostEqual(routing_cost(self.output, self.openai)['score'], 0.000069)
        self.openai['parameters'] = {'prompt_cache_options': {'ttl': '12h'}}
        self.assertAlmostEqual(routing_cost(self.output, self.openai)['score'], 0.000094)

    def test_haiku_prices_output_including_thinking_once(self):
        config = {'provider': 'anthropic', 'model': 'claude-haiku-5-5'}
        output = {**self.output, 'usage': {'input_tokens': 1000, 'output_tokens': 100,
            'cache_read_input_tokens': 0, 'cache_creation_input_tokens': 0,
            'output_tokens_details': {'thinking_tokens': 85}}}
        result = routing_cost(output, config)
        self.assertAlmostEqual(result['score'], 0.00015)
        self.assertEqual(result['metadata']['pricing_date'], '2026-11-02')
        total = make_cost_evaluators(config)[1](output, self.reference)
        self.assertAlmostEqual(total['score'], 0.00115)
        self.assertEqual(self.reference['intern_cost'], 0.001)

    def test_haiku_cache_tokens_determine_long_context_rate(self):
        config = {'provider': 'anthropic', 'model': 'claude-haiku-5-5'}
        usage = {'input_tokens': 90000, 'output_tokens': 100,
                 'cache_read_input_tokens': 9000, 'cache_creation_input_tokens': 1000,
                 'cache_creation': {'ephemeral_5m_input_tokens': 400, 'ephemeral_1h_input_tokens': 600}}
        output = {**self.output, 'usage': usage}
        self.assertAlmostEqual(routing_cost(output, config)['score'], 0.00931)
        usage['cache_read_input_tokens'] += 1
        self.assertAlmostEqual(routing_cost(output, config)['score'], 0.00931001 * 5)
        usage['cache_creation']['ephemeral_1h_input_tokens'] = 500
        with self.assertRaises(ValueError):
            routing_cost(output, config)
        usage['cache_creation_input_tokens'] = 0
        usage['inference_geo'] = 'us'
        with self.assertRaises(ValueError):
            routing_cost(output, config)

    def test_long_context_multiplier_and_zero_usage(self):
        self.output['usage'] = {'input_tokens': 272_000, 'input_tokens_details': {
            'cached_tokens': 0, 'cache_write_tokens': 0}}
        self.assertAlmostEqual(routing_cost(self.output, self.openai)['score'], 0.0272)
        self.output['usage']['input_tokens'] = 300_000
        self.assertAlmostEqual(routing_cost(self.output, self.openai)['score'], 0.06)
        self.output['usage']['input_tokens'] = 0
        self.assertEqual(routing_cost(self.output, self.openai)['score'], 0)

    def test_missing_invalid_or_unknown_usage_is_not_free(self):
        for usage in (None, {}, {'input_tokens': -1}, {'input_tokens': True},
                      {'input_tokens': float('nan')}, {'input_tokens': float('inf')}):
            with self.subTest(usage=usage), self.assertRaises(ValueError):
                routing_cost({**self.output, 'usage': usage}, self.jev)
        with self.assertRaises(ValueError):
            routing_cost(self.output, {**self.jev, 'model': 'unknown'})
        with self.assertRaises(ValueError):
            routing_cost(self.output, self.openai)
        self.output['usage']['input_tokens_details'] = {'cached_tokens': 900, 'cache_write_tokens': 200}
        with self.assertRaises(ValueError):
            routing_cost(self.output, self.openai)

    def test_total_preserves_published_cost_and_invalid_usage_flag(self):
        total = make_cost_evaluators(self.jev)[1]
        self.reference['gpt5_completion_tokens'] = -1
        result = total(self.output, self.reference)
        self.assertAlmostEqual(result['score'], 0.001042)
        self.assertTrue(result['metadata']['invalid_usage'])
        self.assertEqual(self.reference['intern_cost'], 0.001)

    def test_rejects_custom_endpoints_with_unverified_pricing(self):
        for config in (self.jev, self.openai, {'provider': 'anthropic', 'model': 'claude-haiku-5-5'}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                routing_cost(self.output, {**config, 'base_url': 'https://other.invalid/v1'})

    def test_gap_uses_question_specific_active_max_including_partial_scores(self):
        self.reference['gemini_pro_score'] = 1
        self.reference['flash_score'] = 0.5
        self.assertEqual(gap(self.output, self.reference)['score'], 0)
        self.output.update(decision='Candidate C', selected_model='gpt-5')
        self.assertEqual(gap(self.output, self.reference)['score'], 0.5)
        self.reference.update(intern_score=0, flash_score=0)
        self.assertEqual(gap(self.output, self.reference)['score'], 0)

    def test_gap_rejects_incomplete_or_invalid_scores(self):
        for value in (None, True, -1, 1.1, float('nan')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                gap(self.output, {**self.reference, 'flash_score': value})
        with self.assertRaises(ValueError):
            gap({**self.output, 'selected_model': 'gpt-5'}, self.reference)

    def test_experiment_uses_saved_router_config_and_independent_evaluators(self):
        experiments = Mock()
        experiments.get_experiment.return_value = {'experiment_metadata': {'router_config': self.jev}}
        evaluate_experiment(SimpleNamespace(experiments=experiments), 'experiment-id')
        evaluators = experiments.evaluate_experiment.call_args.kwargs['evaluators']
        self.assertEqual(len(evaluators), 3)
        self.assertAlmostEqual(evaluators[0](self.output)['score'], 0.000042)
        # Gap still works when usage is absent, independently of cost failures.
        self.output['usage'] = None
        self.assertEqual(evaluators[2](self.output, self.reference)['score'], 0.5)


if __name__ == '__main__':
    unittest.main()
