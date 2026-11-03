import contextlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from scripts import load_benchmarks as loader
from routing.benchmark_data import iter_records, INVALID_USAGE_QUERIES


class BenchmarkImportTests(unittest.TestCase):
    def run_import(self, change=None):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root / 'source.tar.gz'
            with tarfile.open(archive, 'w:gz') as bundle:
                for model in ('a', 'b'):
                    data = dict(dataset_name='example', split='test', model_name=model,
                                counts=1, records=[dict(index=0, origin_query='question', prompt='question',
                                prediction='answer', ground_truth='answer', score=1,
                                prompt_tokens=2, completion_tokens=1, cost=0.1)])
                    if change and change(model, data) is False:
                        continue
                    raw = json.dumps(data).encode()
                    member = tarfile.TarInfo(f'bench-release/example/test/{model}/result-20260101_000000.json')
                    member.size = len(raw)
                    bundle.addfile(member, io.BytesIO(raw))
            with patch.object(loader, 'MODELS', ('a', 'b')), patch.object(loader, 'BENCHMARKS', {'example': ('test', 1)}), contextlib.redirect_stdout(io.StringIO()):
                report = loader.import_archive(archive, root / 'out')
            if report['complete']:
                self.assertEqual(len(list(iter_records(root / 'out'))), 2)
            else:
                with self.assertRaises(ValueError):
                    list(iter_records(root / 'out'))
            return report

    def test_complete(self):
        self.assertTrue(self.run_import()['complete'])

    def test_missing_model(self):
        self.assertFalse(self.run_import(lambda model, data: model != 'b')['complete'])

    def test_missing_cost(self):
        def change(model, data):
            data['records'][0].pop('cost')
        report = self.run_import(change)
        self.assertFalse(report['complete'])
        self.assertEqual(report['coverage'][0]['errors']['missing_cost'], 1)

    def test_query_mismatch(self):
        def change(model, data):
            if model == 'b':
                data['records'][0]['origin_query'] = 'different question'
        self.assertFalse(self.run_import(change)['complete'])

    def test_duplicate_indices(self):
        def change(model, data):
            data['records'] *= 2
            data['counts'] = 2
        report = self.run_import(change)
        self.assertEqual(report['coverage'][0]['errors']['duplicate_indices'], 1)

    def test_invalid_usage_queries_excluded_for_every_model(self):
        import gzip
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'completeness.json').write_text(json.dumps({'complete': True}))
            key = next(iter(INVALID_USAGE_QUERIES))
            rows = [dict(dataset_id=key[0], split=key[1], record_index=key[2], model_name=m)
                    for m in ('a', 'b')]
            rows.append(dict(dataset_id=key[0], split=key[1], record_index=-1, model_name='a'))
            with gzip.open(root / 'records.jsonl.gz', 'wt') as stream:
                for row in rows:
                    stream.write(json.dumps(row) + '\n')
            self.assertEqual(len(list(iter_records(root))), 1)
            self.assertEqual(len(list(iter_records(root, exclude_invalid_usage=False))), 3)


if __name__ == '__main__':
    unittest.main()
