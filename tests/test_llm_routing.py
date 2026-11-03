import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from anthropic import Anthropic
import httpx2
from pydantic import ValidationError

from routing.llm_routing.router import HaikuClassifier, RoutingDecision
from routing.pipeline import run_pipeline
from routing.prompts import CANDIDATES, PROMPT_OBJECTIVES, build_system_prompt
from routing.router import ModelRouter


class HaikuRoutingTests(unittest.TestCase):
    def test_pydantic_schema_allows_exactly_the_shared_candidates(self):
        schema = RoutingDecision.model_json_schema()
        self.assertEqual(schema['properties']['decision']['enum'], list(CANDIDATES))
        self.assertFalse(schema['additionalProperties'])
        for invalid in ({'decision': 'Candidate D'}, {'decision': 'gpt-5'},
                        {'decision': 'Candidate A', 'reason': 'extra'}, {}):
            with self.subTest(invalid=invalid), self.assertRaises(ValidationError):
                RoutingDecision.model_validate(invalid)

    def test_actual_sdk_sends_shared_prompt_and_constrained_schema(self):
        requests = []

        def respond(request):
            payload = json.loads(request.content)
            requests.append(payload)
            return httpx2.Response(200, json={
                'id': 'msg_test', 'type': 'message', 'role': 'assistant',
                'model': 'claude-haiku-5-5', 'stop_reason': 'end_turn',
                'stop_sequence': None,
                'content': [{'type': 'text', 'text': '{"decision":"Candidate B"}'}],
                'usage': {'input_tokens': 35, 'output_tokens': 8},
            })

        with httpx2.Client(transport=httpx2.MockTransport(respond)) as http_client:
            client = Anthropic(api_key='test-key', http_client=http_client, max_retries=0)
            with patch('routing.llm_routing.router.Anthropic', return_value=client):
                for variant in ('default', *PROMPT_OBJECTIVES):
                    instructions = build_system_prompt(variant=variant)
                    router = ModelRouter(HaikuClassifier(
                        model='claude-haiku-5-5', system_prompt=instructions, api_key='test-key'))
                    decision, usage = router.route('question')
                    self.assertEqual(decision, 'Candidate B')
                    self.assertEqual(usage['input_tokens'], 35)
                    self.assertEqual(usage['output_tokens'], 8)
                    payload = requests[-1]
                    self.assertEqual(payload['system'], instructions)
                    self.assertEqual(payload['messages'], [{'role': 'user', 'content': 'question'}])
                    output_format = payload['output_config']['format']
                    self.assertEqual(output_format['type'], 'json_schema')
                    self.assertEqual(output_format['schema']['properties']['decision']['enum'],
                                     list(CANDIDATES))
                    for identity in ('intern-s1', 'gemini-2.5-flash', 'gpt-5'):
                        self.assertNotIn(identity, json.dumps(payload))

    def test_incomplete_or_refused_response_has_no_fallback(self):
        with patch('routing.llm_routing.router.Anthropic') as factory:
            router = ModelRouter(HaikuClassifier(
                model='claude-haiku-5-5', system_prompt=build_system_prompt(), api_key='key'))
            for stop_reason, parsed in (('refusal', None), ('max_tokens', None),
                                        ('end_turn', None)):
                factory.return_value.messages.parse.return_value = SimpleNamespace(
                    stop_reason=stop_reason, parsed_output=parsed)
                with self.subTest(stop_reason=stop_reason), self.assertRaises(ValueError):
                    router.route('question')

    def test_parameters_cannot_override_prompt_or_output_schema(self):
        for key in ('model', 'system', 'messages', 'output_format', 'output_config', 'extra_body'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                HaikuClassifier(model='test', system_prompt='shared', api_key='key',
                                parameters={key: 'override'})

    def test_pipeline_preserves_mapping_usage_and_excludes_reference_data(self):
        config = json.loads((Path(__file__).resolve().parents[1] /
                             'routing/llm_routing/config.json').read_text())

        class Dataset(list):
            id = 'dataset-id'
            version_id = config['dataset_version']

        dataset = Dataset([{'id': 'e1', 'input': {'prompt': 'question', 'extra': 'hidden'},
                            'output': {'solver': 'hidden'}, 'metadata': {'secret': 'hidden'}}])
        client = SimpleNamespace(datasets=Mock(), experiments=Mock())
        client.datasets.get_dataset.return_value = dataset
        client.experiments.run_experiment.side_effect = lambda **kwargs: [
            kwargs['task'](example['input'], example) for example in kwargs['dataset']]
        with TemporaryDirectory() as folder, patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'key'}), patch(
                'routing.llm_routing.router.Anthropic') as factory:
            factory.return_value.messages.parse.return_value = SimpleNamespace(
                stop_reason='end_turn', parsed_output=RoutingDecision(decision='Candidate C'),
                usage=SimpleNamespace(model_dump=lambda **kwargs: {'input_tokens': 11, 'output_tokens': 8}))
            output = Path(folder) / 'decisions.jsonl'
            result = run_pipeline(config, output, router_name='haiku', client=client)
            self.assertEqual(result['haiku'][0]['selected_model'], 'gpt-5')
            record = json.loads(output.read_text())
            self.assertEqual(record['decision'], 'Candidate C')
            self.assertEqual(record['usage'], {'input_tokens': 11, 'output_tokens': 8})
            self.assertEqual(record['dataset_version'], config['dataset_version'])
            request = factory.return_value.messages.parse.call_args.kwargs
            self.assertEqual(request['system'], build_system_prompt())
            self.assertEqual(request['messages'], [{'role': 'user', 'content': 'question'}])
            self.assertIs(request['output_format'], RoutingDecision)
            factory.assert_called_once_with(api_key='key', base_url='https://api.anthropic.com',
                                            timeout=60, max_retries=0)


if __name__ == '__main__':
    unittest.main()
