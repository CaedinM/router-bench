import json
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError

from routing.pipeline import make_phoenix_client, make_task, run_pipeline
from routing.prompts import CANDIDATES, DEFAULT_PROFILES, PROMPT_OBJECTIVES, build_system_prompt
from routing.providers.jev import JevClassifier
from routing.providers.http import ProviderHTTPError, post_json
from routing.providers.openai_decisions import OpenAIDecisionsClassifier
from routing.router import ModelRouter


class RoutingTests(unittest.TestCase):
    def test_objective_variants_preserve_anonymous_profiles_and_default(self):
        default = build_system_prompt()
        variants = [build_system_prompt(variant=variant) for variant in PROMPT_OBJECTIVES]
        self.assertEqual(len(set([default, *variants])), 4)
        for variant, prompt in zip(PROMPT_OBJECTIVES, variants):
            self.assertIn(PROMPT_OBJECTIVES[variant], prompt)
            for label in CANDIDATES:
                self.assertIn(f'{label}: {DEFAULT_PROFILES[label]}', prompt)
            for identity in ('intern-s1', 'gemini-2.5-flash', 'gpt-5'):
                self.assertNotIn(identity, prompt.lower())
        with self.assertRaises(ValueError):
            build_system_prompt(variant='typo')

    def test_candidate_mapping_is_saved_after_the_provider_decision(self):
        expected_models = {
            'Candidate A': 'intern-s1',
            'Candidate B': 'gemini-2.5-flash',
            'Candidate C': 'gpt-5',
        }
        for label, model_id in expected_models.items():
            response = {'answers': {'route': {'choice': label}}, 'usage': {'input_tokens': 1}}
            with self.subTest(label=label), TemporaryDirectory() as folder, patch(
                    'routing.providers.jev.post_json', return_value=response) as post:
                classifier = JevClassifier(model='routing-model', system_prompt='anonymous', api_key='key')
                output = Path(folder) / 'results.jsonl'
                result = make_task(ModelRouter(classifier), 'jev', output, {})({'prompt': 'q'}, {'id': 'e1'})
                self.assertEqual(result['decision'], label)
                self.assertEqual(result['selected_model'], model_id)
                self.assertEqual(json.loads(output.read_text())['selected_model'], model_id)
                request = json.dumps(post.call_args.args[1])
                for candidate_model in expected_models.values():
                    self.assertNotIn(candidate_model, request)

    def test_jev_retries_long_context_error_without_an_extra_task_record(self):
        error_body = io.BytesIO(b'{"detail":{"error_type":"max_tokens_exceeded"}}')
        error = HTTPError('https://provider.invalid', 400, 'Bad Request', {}, error_body)
        response = io.BytesIO(b'{"answers":{"route":{"choice":"Candidate B"}},"usage":{"input_tokens":99}}')
        classifier = JevClassifier(model='test', system_prompt='shared', api_key='secret')
        with TemporaryDirectory() as folder, patch('routing.providers.http.urlopen',
                                                    side_effect=[error, response]) as urlopen:
            output = Path(folder) / 'decisions.jsonl'
            task = make_task(ModelRouter(classifier), 'jev', output, {})
            result = task({'prompt': 'long question'}, {'id': 'e1'})
            self.assertEqual(result['decision'], 'Candidate B')
            self.assertEqual(result['usage'], {'input_tokens': 99})
            records = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual([record['status'] for record in records], ['ok'])
            self.assertEqual(records[0]['usage'], {'input_tokens': 99})
            first_request = urlopen.call_args_list[0].args[0]
            retry_request = urlopen.call_args_list[1].args[0]
            self.assertIsNone(first_request.get_header('Typesafe-danger-zone-long-context'))
            self.assertEqual(retry_request.get_header('Typesafe-danger-zone-long-context'), 'true')
            self.assertEqual(first_request.data, retry_request.data)

    def test_jev_only_retries_token_limit_error_once(self):
        classifier = JevClassifier(model='test', system_prompt='shared', api_key='secret')
        for status, error_type in ((400, 'other_error'), (429, 'max_tokens_exceeded')):
            with self.subTest(status=status, error_type=error_type), patch(
                    'routing.providers.jev.post_json',
                    side_effect=ProviderHTTPError(status, 'provider error', error_type)) as post:
                with self.assertRaises(ProviderHTTPError):
                    classifier.predict('question')
                self.assertEqual(post.call_count, 1)
        with patch('routing.providers.jev.post_json', side_effect=ProviderHTTPError(
                400, 'provider error', 'max_tokens_exceeded')) as post:
            with self.assertRaises(ProviderHTTPError):
                classifier.predict('question')
            self.assertEqual(post.call_count, 2)

    def test_http_error_includes_provider_response_without_api_key(self):
        response = io.BytesIO(b'{"error": "request too large", "token": "test-secret"}')
        error = HTTPError('https://provider.invalid', 400, 'Bad Request', {}, response)
        with patch('routing.providers.http.urlopen', side_effect=error):
            with self.assertRaises(RuntimeError) as caught:
                post_json('https://provider.invalid', {}, {'Authorization': 'Bearer test-secret'}, 1)
        self.assertIn('request too large', str(caught.exception))
        self.assertNotIn('test-secret', str(caught.exception))

    def test_phoenix_url_resolves_environment_reference(self):
        client_factory = Mock()
        with patch.dict('sys.modules', {'phoenix.client': SimpleNamespace(Client=client_factory)}), patch.dict(
                'os.environ', {'PHOENIX_BASE_URL': 'https://phoenix.example.test'}):
            client = make_phoenix_client({'phoenix_url': 'PHOENIX_BASE_URL'})
            client_factory.assert_called_once_with(base_url='https://phoenix.example.test')
            self.assertIs(client, client_factory.return_value)

    def test_phoenix_url_still_accepts_literal_url(self):
        client_factory = Mock()
        with patch.dict('sys.modules', {'phoenix.client': SimpleNamespace(Client=client_factory)}):
            make_phoenix_client({'phoenix_url': 'http://127.0.0.1:6006'})
            client_factory.assert_called_once_with(base_url='http://127.0.0.1:6006')

    def test_async_phoenix_client_uses_the_same_url_and_full_dataset_timeout(self):
        client_factory = Mock()
        with patch.dict('sys.modules', {'phoenix.client': SimpleNamespace(AsyncClient=client_factory)}):
            client = make_phoenix_client({'phoenix_url': 'http://127.0.0.1:6006'}, asynchronous=True)
            client_factory.assert_called_once_with(base_url='http://127.0.0.1:6006')
            self.assertEqual(client._client.timeout.read, 600)

    def test_missing_or_invalid_phoenix_environment_url_fails(self):
        for value in ('', 'not-a-url'):
            with self.subTest(value=value), patch.dict('os.environ', {'PHOENIX_BASE_URL': value}):
                with self.assertRaisesRegex(ValueError, 'PHOENIX_BASE_URL'):
                    make_phoenix_client({'phoenix_url': 'PHOENIX_BASE_URL'})

    def test_both_providers_share_instructions_prompt_and_options(self):
        instructions = build_system_prompt()
        adapters = [
            ('jev', JevClassifier, {'answers': {'route': {'choice': 'Candidate B'}}}),
            ('openai_decisions', OpenAIDecisionsClassifier,
             {'answers': [{'type': 'choice', 'name': 'route', 'choice': 'Candidate B'}]}),
        ]
        for provider, cls, response in adapters:
            with self.subTest(provider=provider), patch(
                    f'routing.providers.{provider}.post_json', return_value=response) as post:
                router = ModelRouter(cls(model='test', system_prompt=instructions, api_key='secret'))
                self.assertEqual(router.route('question'), ('Candidate B', None))
                payload = post.call_args.args[1]
                question = payload['questions']['route'] if provider == 'jev' else payload['questions'][0]
                self.assertEqual(question['instructions'], instructions)
                self.assertEqual(payload.get('state', payload.get('input')), 'question')
                labels = tuple(question['criteria']) if provider == 'jev' else tuple(
                    choice['value'] for choice in question['choices'])
                self.assertEqual(labels, CANDIDATES)
                self.assertNotIn('secret', json.dumps(payload))

    def test_cannot_override_shared_contract(self):
        for cls, key in ((JevClassifier, 'questions'), (OpenAIDecisionsClassifier, 'input')):
            with self.assertRaises(ValueError):
                cls(model='test', system_prompt='shared', api_key='key', parameters={key: 'override'})

    def test_provider_usage_is_saved_with_decision(self):
        usage = {'input_tokens': 23, 'output_tokens': 1,
                 'input_tokens_details': {'cached_tokens': 7}}
        adapters = [
            ('jev', JevClassifier, {'answers': {'route': {'choice': 'Candidate B'}}, 'usage': usage}),
            ('openai_decisions', OpenAIDecisionsClassifier,
             {'answers': [{'type': 'choice', 'name': 'route', 'choice': 'Candidate B'}], 'usage': usage}),
        ]
        for provider, classifier_class, response in adapters:
            with self.subTest(provider=provider), TemporaryDirectory() as folder, patch(
                    f'routing.providers.{provider}.post_json', return_value=response):
                router = ModelRouter(classifier_class(model='test', system_prompt='shared', api_key='key'))
                output = Path(folder) / 'decisions.jsonl'
                task = make_task(router, provider, output, {})
                result = task({'prompt': 'question'}, {'id': 'e1'})
                self.assertEqual(result['usage'], usage)
                self.assertEqual(json.loads(output.read_text())['usage'], usage)

    def test_missing_provider_usage_is_null(self):
        classifier = JevClassifier(model='test', system_prompt='shared', api_key='key')
        with patch('routing.providers.jev.post_json', return_value={
                'answers': {'route': {'choice': 'Candidate A'}}}):
            self.assertEqual(ModelRouter(classifier).route('question'), ('Candidate A', None))

    def test_router_validates_decisions_and_preserves_legacy_mapping(self):
        classifier = Mock()
        classifier.predict.return_value = ' Candidate C\n'
        self.assertEqual(ModelRouter(classifier).route('question'), ('Candidate C', None))
        classifier.predict.return_value = 'Candidate D'
        with self.assertRaises(ValueError):
            ModelRouter(classifier).route('question')
        classifier.predict.return_value = 'easy'
        self.assertEqual(ModelRouter(classifier, {'easy': 'legacy'}).route('question'), ('legacy', None))

    def test_failure_saved_without_fallback_and_next_example_runs(self):
        router = Mock()
        router.route_with_usage.side_effect = [ValueError('invalid'), ('Candidate A', None)]
        with TemporaryDirectory() as folder:
            output = Path(folder) / 'decisions.jsonl'
            task = make_task(router, 'router', output, {'run_id': 'run'})
            with self.assertRaises(ValueError):
                task({'prompt': 'q1'}, {'id': 'e1'})
            self.assertEqual(task({'prompt': 'q2'}, {'id': 'e2'})['decision'], 'Candidate A')
            records = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual([r['status'] for r in records], ['error', 'ok'])
            self.assertIsNone(records[0]['decision'])
            self.assertIsNone(records[0]['selected_model'])

    def test_only_selected_router_runs_and_reference_does_not_leak(self):
        examples = [dict(id='e1', input={'prompt': 'q1', 'extra': 'hidden'},
                         output={'model_name': 'hidden solver'}, metadata={'secret': 'hidden'})]

        class Dataset(list):
            id = 'dataset-id'
            version_id = 'version-id'

        dataset = Dataset(examples)
        config = {'routers': [
            dict(name='jev', provider='jev', model='test'),
            dict(name='openai', provider='openai_decisions', model='test'),
        ]}
        # Anonymous profiles need no calibration configuration or separate process.
        config['candidate_profiles'] = dict.fromkeys(CANDIDATES, 'Public-source profile')
        config['system_prompt_variant'] = 'accuracy-first'
        client = SimpleNamespace(datasets=Mock(), experiments=Mock())
        client.datasets.get_dataset.return_value = dataset
        calls = []

        def run(**kwargs):
            calls.append(kwargs)
            return [kwargs['task'](item['input'], item) for item in kwargs['dataset']]

        client.experiments.run_experiment.side_effect = run
        with TemporaryDirectory() as folder, patch.dict('os.environ', {'TYPESAFE_API_KEY': 'test-key'}), patch(
                'routing.providers.jev.post_json', return_value={
                    'answers': {'route': {'choice': 'Candidate B'}}}) as post:
            output = Path(folder) / 'results.jsonl'
            results = run_pipeline(config, output, router_name='jev', client=client)
            self.assertEqual(list(results), ['jev'])
            self.assertEqual(len(calls), 1)
            self.assertNotIn('evaluators', calls[0])
            self.assertIn(PROMPT_OBJECTIVES['accuracy-first'],
                          calls[0]['experiment_metadata']['system_prompt'])
            self.assertEqual(calls[0]['experiment_metadata']['system_prompt_variant'], 'accuracy-first')
            self.assertEqual(calls[0]['retries'], 1)
            payload = post.call_args.args[1]
            self.assertEqual(payload['state'], 'q1')
            self.assertNotIn('hidden', json.dumps(payload))
            record = json.loads(output.read_text())
            self.assertEqual(record['example_id'], 'e1')
            self.assertEqual(record['dataset_version'], 'version-id')
            self.assertEqual(record['system_prompt_variant'], 'accuracy-first')
            client.datasets.get_dataset.assert_called_once_with(
                dataset='router-bench-intern-flash-gpt5-full-v1', version_id=None, splits=['development'], timeout=600)
            # A separate invocation for a new router does not instantiate or call Jev.
            with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}), patch(
                    'routing.providers.openai_decisions.post_json', return_value={
                        'answers': [{'type': 'choice', 'name': 'route', 'choice': 'Candidate C'}]}):
                run_pipeline(config, output, router_name='openai', client=client)
            self.assertEqual(post.call_count, 1)
            records = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual([r['router'] for r in records], ['jev', 'openai'])
            self.assertEqual(records[0]['system_prompt_sha256'], records[1]['system_prompt_sha256'])

    def test_profile_validation(self):
        with self.assertRaises(ValueError):
            build_system_prompt({'Candidate A': 'profile'})
        with self.assertRaises(ValueError):
            build_system_prompt({})


if __name__ == '__main__':
    unittest.main()
