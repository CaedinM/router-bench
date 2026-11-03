"""Exercise notebook cells through the existing uploader and pipeline, without APIs."""
import ast
import contextlib
import inspect
import hashlib
import io
import json
import os
import logging
import warnings
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from routing import pipeline

real_load_environment = pipeline.load_environment

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'notebooks/run_routing_experiments.ipynb'
from routing import upload_dataset as uploader


class Dataset(list):
    id = 'dataset-id'
    version_id = 'fresh-version'


class RoutingNotebookTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.notebook = json.loads(NOTEBOOK.read_text())
        self.cells = [''.join(cell['source']) for cell in self.notebook['cells']
                      if cell['cell_type'] == 'code']
        self.state = {}
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.dataset = Dataset([{'id': 'example', 'input': {'prompt': 'Question only'},
                                 'output': {'private': 'reference'}, 'metadata': {'private': True}}])
        self.prompts = []
        prompts = self.prompts

        class Classifier:
            def __init__(self, **kwargs):
                self.options = kwargs

            def predict(self, prompt):
                prompts.append(prompt)
                return 'Candidate A'

        self.provider = SimpleNamespace(JevClassifier=Classifier,
                                        OpenAIDecisionsClassifier=Classifier, HaikuClassifier=Classifier)
        self.calls = []

        async def experiment(**kwargs):
            self.calls.append(kwargs)
            for example in kwargs['dataset']:
                await kwargs['task'](example['input'], example)
            return {'experiment_id': f'experiment-{len(self.calls)}'}

        self.async_client = SimpleNamespace(
            datasets=SimpleNamespace(get_dataset=AsyncMock(return_value=self.dataset)),
            experiments=SimpleNamespace(run_experiment=AsyncMock(side_effect=experiment)),
            _client=SimpleNamespace(aclose=AsyncMock()))
        self.sync_client = SimpleNamespace(
            datasets=SimpleNamespace(list=Mock(return_value=[{'name': uploader.NAME}])),
            _client=SimpleNamespace(close=Mock(), get=Mock(return_value=Mock(json=Mock(return_value={
                'data': [{'name': f'{partition}/{benchmark}'} for partition in ('development', 'test')
                         for benchmark in pipeline.BENCHMARKS], 'next_cursor': None}))))
        )
        self.fake_phoenix = SimpleNamespace(active_session=Mock(return_value=None), launch_app=Mock())
        for patcher in (
            patch('routing.pipeline.load_environment'),
            patch('routing.upload_dataset.load_environment'),
            patch.dict(os.environ, {'TYPESAFE_API_KEY': 'test', 'OPENAI_API_KEY': 'test',
                                    'ANTHROPIC_API_KEY': 'test'}),
            patch('routing.pipeline.make_phoenix_client', return_value=self.async_client),
            patch('routing.upload_dataset.make_phoenix_client', return_value=self.sync_client),
            patch.dict(sys.modules, {'phoenix': self.fake_phoenix}),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        await self.execute(0)

    async def execute(self, index):
        compiled = compile(self.cells[index], str(NOTEBOOK), 'exec',
                           flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        with contextlib.redirect_stdout(io.StringIO()):
            result = eval(compiled, self.state)
            if inspect.isawaitable(result):
                await result

    async def initialize(self, selected=None, variant='default', custom_prompt=None):
        await self.execute(3)
        if selected is not None:
            self.state['selected_routers'] = selected
        await self.execute(4)
        self.state.update(system_prompt_variant=variant, custom_system_prompt=custom_prompt)
        with patch('importlib.import_module', return_value=self.provider):
            await self.execute(5)

    async def run_experiments(self):
        self.state.setdefault('benchmarks_filter', None)
        # Keep the default per-run logs in a temporary directory during testing.
        with patch('routing.pipeline.__file__', str(Path(self.temp.name) / 'pipeline.py')):
            await self.execute(7)

    async def test_session_setup_uses_installed_imports_without_path_changes(self):
        self.assertNotIn('sys.path', self.cells[0])
        self.assertNotIn('Path.cwd', self.cells[0])
        self.assertEqual(self.state['upload_dataset'].__module__, 'routing.upload_dataset')
        setup = ''.join(self.notebook['cells'][0]['source'])
        self.assertIn('pip install -e ".[notebook]"', setup)
        self.assertIn('One-time setup', setup)
        self.assertIn('Each session', setup)

    async def test_notebook_warning_filters_leave_unrelated_messages_visible(self):
        from pydantic.json_schema import PydanticJsonSchemaWarning
        from sqlalchemy.exc import SAWarning
        logger = logging.getLogger('phoenix.server.app')
        original_filters = logger.filters[:]
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                real_load_environment(quiet_phoenix=True)
                warnings.warn('Column-expression-level unary distinct() should not be used outside '
                              'of an aggregate function. For general SELECT DISTINCT support use select().distinct().',
                              SAWarning)
                warnings.warn_explicit(
                    'Default value <phoenix.db.types.db_helper_types.Undefined object at 0x123> '
                    'is not JSON serializable; excluding default from JSON schema [non-serializable-default]',
                    PydanticJsonSchemaWarning, 'json_schema.py', 1, module='pydantic.json_schema')
                warnings.warn('Another SQLAlchemy warning', SAWarning)
                warnings.warn('Another schema warning', PydanticJsonSchemaWarning)
                self.assertEqual([str(w.message) for w in caught],
                                 ['Another SQLAlchemy warning', 'Another schema warning'])
            known = logger.makeRecord(logger.name, logging.WARNING, '', 1,
                'boto3 is installed but aioboto3 is not. To use AWS Bedrock models in Playground, install aioboto3',
                (), None)
            other = logger.makeRecord(logger.name, logging.ERROR, '', 1, 'Server failure', (), None)
            self.assertFalse(logger.filter(known))
            self.assertTrue(logger.filter(other))
        finally:
            logger.filters[:] = original_filters

    async def test_notebook_all_routers_shared_version_unique_logs_and_no_config_reads(self):
        await self.execute(1)
        self.fake_phoenix.launch_app.assert_called_once_with(
            use_temp_dir=False)
        await self.execute(2)
        self.sync_client._client.close.assert_called_once()
        # The notebook path must not read CLI configs, even during initialization/run.
        with patch('pathlib.Path.read_text', side_effect=AssertionError('No config files')):
            await self.initialize()
            await self.execute(6)
            for _ in range(2):
                await self.run_experiments()
        self.assertEqual(len(self.calls), 6)
        self.assertEqual(set(self.state['results']), {'jev', 'openai-decisions', 'haiku'})
        for call in self.calls:
            self.assertIs(call['dataset'], self.dataset)
            self.assertEqual(call['concurrency'], 16)
            self.assertEqual(call['experiment_metadata']['dataset_version'], 'fresh-version')
            self.assertEqual(call['experiment_metadata']['split'], 'development')
        self.assertEqual(self.async_client.datasets.get_dataset.await_count, 2)
        logs = list(Path(self.temp.name).glob('results/*.jsonl'))
        self.assertEqual(len(logs), 6)
        for log in logs:
            record = json.loads(log.read_text())
            self.assertEqual(record['decision'], 'Candidate A')
            self.assertEqual(record['selected_model'], 'intern-s1')
        self.assertEqual(self.prompts, ['Question only'] * 6)
        self.assertEqual(self.async_client._client.aclose.await_count, 2)
        self.assertEqual(len({router.system_prompt for router in self.state['routers']}), 1)

    async def test_prompt_variants_and_custom_override_reach_all_routers_and_metadata(self):
        from routing.prompts import build_system_prompt, DEFAULT_PROFILES

        for variant, custom in (
            ('default', None), ('balanced', None), ('accuracy-first', None), ('cost-first', None),
            ('cost-first', 'Select the best candidate for this request. Return only its candidate label.'),
        ):
            with self.subTest(variant=variant, custom=custom):
                await self.initialize(variant=variant, custom_prompt=custom)
                prompt = build_system_prompt(variant=variant, custom_prompt=custom)
                for router in self.state['routers']:
                    self.assertEqual(router.classifier.options['system_prompt'], prompt)
                    for label, description in DEFAULT_PROFILES.items():
                        self.assertIn(f'{label}: {description}', prompt)
                self.state['split'] = 'development'
                await self.run_experiments()
                for call in self.calls[-3:]:
                    metadata = call['experiment_metadata']
                    self.assertEqual(metadata['system_prompt_variant'], 'custom' if custom else variant)
                    self.assertEqual(metadata['system_prompt'], prompt)
                    self.assertEqual(metadata['system_prompt_sha256'], hashlib.sha256(prompt.encode()).hexdigest())
                if custom:
                    self.assertTrue(prompt.startswith(custom))

    async def test_invalid_prompt_settings_fail_during_initialization(self):
        for variant, custom in (('unknown', None), ('default', ''), ('default', '   ')):
            with self.subTest(variant=variant, custom=custom):
                with self.assertRaises(ValueError):
                    await self.initialize(variant=variant, custom_prompt=custom)
                self.assertEqual(self.state['routers'], [])
        self.async_client.experiments.run_experiment.assert_not_awaited()

    async def test_subset_uses_injected_attributes_and_selected_split(self):
        await self.initialize(['haiku'])
        self.state['split'] = 'test'
        await self.run_experiments()
        self.assertEqual(list(self.state['results']), ['haiku'])
        self.assertEqual(self.calls[0]['experiment_metadata']['router_config']['model'], 'claude-haiku-5-5')
        self.async_client.datasets.get_dataset.assert_awaited_once_with(
            dataset=uploader.NAME, version_id=None, splits=['test'], timeout=600)

    async def test_benchmark_subset_selects_partition_specific_splits_and_records_selection(self):
        await self.initialize(['jev'])
        self.state.update(split='development', benchmarks_filter=['aime', 'gpqa'])
        await self.run_experiments()
        self.async_client.datasets.get_dataset.assert_awaited_once_with(
            dataset=uploader.NAME, version_id=None,
            splits=['development/aime', 'development/gpqa'], timeout=600)
        self.assertEqual(self.calls[0]['experiment_metadata']['benchmarks'], ['aime', 'gpqa'])
        record = json.loads(next(Path(self.temp.name).glob('results/*.jsonl')).read_text())
        self.assertEqual(record['benchmarks'], ['aime', 'gpqa'])

    async def test_invalid_benchmarks_fail_before_dataset_fetch_or_routing(self):
        await self.initialize(['jev'])
        self.state['split'] = 'development'
        for selection in ([], ['typo'], ['aime', 'aime'], 'aime', [None]):
            with self.subTest(selection=selection):
                self.state['benchmarks_filter'] = selection
                with self.assertRaises(ValueError):
                    await self.run_experiments()
        self.async_client.datasets.get_dataset.assert_not_awaited()
        self.async_client.experiments.run_experiment.assert_not_awaited()

    async def test_invalid_selection_and_missing_credentials_fail_before_requests(self):
        for selected in ([], ['jev', 'jev']):
            await self.initialize(selected)
            self.state['split'] = 'development'
            with self.assertRaises(ValueError):
                await self.run_experiments()
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'ANTHROPIC_API_KEY'):
                await self.initialize(['haiku'])
        self.async_client.experiments.run_experiment.assert_not_awaited()

    async def test_failure_closes_client_preserving_completed_experiment_and_log(self):
        await self.initialize()
        self.state['split'] = 'development'
        original = self.async_client.experiments.run_experiment.side_effect

        async def fail_second(**kwargs):
            if self.calls:
                raise RuntimeError('experiment failed')
            return await original(**kwargs)

        self.async_client.experiments.run_experiment.side_effect = fail_second
        with self.assertRaisesRegex(RuntimeError, 'experiment failed'):
            await self.run_experiments()
        self.assertEqual(self.async_client.experiments.run_experiment.await_count, 2)
        self.assertEqual(len(list(Path(self.temp.name).glob('results/*.jsonl'))), 1)
        self.async_client._client.aclose.assert_awaited_once()

    async def test_notebook_format_and_launch_directories(self):
        self.assertEqual((self.notebook['nbformat'], self.notebook['nbformat_minor']), (4, 5))
        for cell in self.notebook['cells']:
            if cell['cell_type'] == 'code':
                self.assertIsInstance(cell['outputs'], list)
                self.assertTrue(cell['execution_count'] is None or isinstance(cell['execution_count'], int))
        for cwd in (ROOT, ROOT / 'notebooks'):
            with patch('pathlib.Path.cwd', return_value=cwd):
                await self.execute(0)
            self.assertEqual(self.state['build_router'].__module__, 'routing.pipeline')


class DatasetUploadTests(unittest.TestCase):
    def test_reuse_and_cli_refusal_close_client_without_reading_data(self):
        client = SimpleNamespace(datasets=SimpleNamespace(list=Mock(return_value=[{'name': uploader.NAME}])),
            _client=SimpleNamespace(close=Mock(), get=Mock(return_value=Mock(json=Mock(return_value={
                'data': [{'name': f'{partition}/{benchmark}'} for partition in ('development', 'test')
                         for benchmark in pipeline.BENCHMARKS], 'next_cursor': None}))))
        )
        with patch.object(uploader, 'make_phoenix_client', return_value=client), \
                patch.object(uploader, 'load_environment'), patch.object(uploader.gzip, 'open') as read:
            self.assertEqual(uploader.upload_dataset(), uploader.NAME)
            with self.assertRaisesRegex(ValueError, 'refusing to append'):
                uploader.upload_dataset(reuse_existing=False)
        read.assert_not_called()
        self.assertEqual(client._client.close.call_count, 2)

    def test_existing_dataset_gets_missing_benchmark_splits_without_duplicate_examples(self):
        names = {f'{partition}/{benchmark}' for partition in ('development', 'test')
                 for benchmark in pipeline.BENCHMARKS}
        names.remove('development/aime')
        response = Mock(json=Mock(return_value={
            'data': [{'name': name} for name in names], 'next_cursor': None}))
        datasets = SimpleNamespace(list=Mock(return_value=[{'name': uploader.NAME}]),
            get_dataset=Mock(return_value=[
                {'id': 'aime-example', 'metadata': {'dataset_id': 'aime'}},
                {'id': 'gpqa-example', 'metadata': {'dataset_id': 'gpqa'}}]),
            add_examples_to_dataset=Mock())
        client = SimpleNamespace(datasets=datasets,
            _client=SimpleNamespace(close=Mock(), get=Mock(return_value=response), post=Mock()))
        with patch.object(uploader, 'make_phoenix_client', return_value=client),                 patch.object(uploader, 'load_environment'), patch.object(uploader.gzip, 'open') as read:
            uploader.upload_dataset()
            datasets.get_dataset.assert_called_once_with(dataset=uploader.NAME, splits=['development'], timeout=600)
            client._client.post.assert_called_once_with(
                f'v1/datasets/{uploader.NAME}/splits',
                json={'name': 'development/aime', 'example_ids': ['aime-example']}, timeout=600)
            response.json.return_value['data'].append({'name': 'development/aime'})
            uploader.upload_dataset()
            self.assertEqual(client._client.post.call_count, 1)
        datasets.add_examples_to_dataset.assert_not_called()
        read.assert_not_called()
        self.assertEqual(client._client.close.call_count, 2)

    def test_upload_retains_fixed_splits_and_original_records(self):
        rows = (
            json.dumps({'dataset_id': 'benchmark', 'split': 'test', 'record_index': index,
                        'model_name': model, 'prompt': f'question-{index}', 'origin_query': 'original',
                        'score': 0.5, 'cost': 0.123}) + '\n'
            for index in range(12446) for model in uploader.PREFIX
        )
        datasets = SimpleNamespace(list=Mock(return_value=[]), create_dataset=Mock(),
                                   add_examples_to_dataset=Mock())
        client = SimpleNamespace(datasets=datasets, _client=SimpleNamespace(close=Mock()))
        development = {('benchmark', 'test', index) for index in range(30)}
        with patch.object(uploader, 'make_phoenix_client', return_value=client), \
                patch.object(uploader, 'load_environment'), \
                patch.object(uploader, 'load_development', return_value=development), \
                patch.object(uploader.gzip, 'open', return_value=contextlib.nullcontext(rows)), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(uploader.upload_dataset(), uploader.NAME)
        examples = datasets.create_dataset.call_args.kwargs['examples'][:]
        for call in datasets.add_examples_to_dataset.call_args_list:
            examples.extend(call.kwargs['examples'])
        self.assertEqual(len(examples), 12446)
        self.assertEqual(sum('development' in row['splits'] for row in examples), 30)
        self.assertEqual(sum('test' in row['splits'] for row in examples), 12416)
        for example in examples:
            partition = example['splits'][0]
            self.assertEqual(example['splits'], [partition, f'{partition}/benchmark'])
            for prefix in uploader.PREFIX.values():
                self.assertEqual(example['output'][f'{prefix}_cost'], 0.123)
                self.assertEqual(example['output'][f'{prefix}_score'], 0.5)
        client._client.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
