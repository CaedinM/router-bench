"""Import the published performance-cost subset without running model inference."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import tarfile

MODELS = ('intern-s1', 'gemini-2.5-flash', 'gpt-5')
# Published performance-cost benchmark sizes, not the small-model/subset splits.
BENCHMARKS = {
    'aime': ('hybrid', 60), 'livemathbench': ('test', 121),
    'livecodebench': ('test', 1055), 'swe-bench': ('verified', 500),
    'gpqa': ('test', 198), 'hle': ('test', 2158),
    'mmlupro': ('test_3000', 3000), 'simpleqa': ('test', 4326),
    'arenahard': ('test', 750), 'tau2': ('test', 278),
}
SOURCE = 'https://github.com/ynulihao/LLMRouterBench'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def finite_number(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def import_archive(archive, output, models=None):
    models = tuple(MODELS if models is None else models)
    if not models or len(set(models)) != len(models):
        raise ValueError('Select distinct model identifiers')
    output.mkdir(parents=True, exist_ok=True)
    selected = {}
    # Stream regular JSON members only; never extract archive paths or execute contents.
    with tarfile.open(archive, 'r|gz') as bundle:
        for member in bundle:
            parts = member.name.split('/')
            if not (member.isfile() and member.name.endswith('.json') and len(parts) > 2
                    and parts[1] in BENCHMARKS and set(parts) & set(models)):
                continue
            raw = bundle.extractfile(member).read()
            data = json.loads(raw)
            dataset, split, model = (data.get(k) for k in ('dataset_name', 'split', 'model_name'))
            if dataset not in BENCHMARKS or model not in models or data.get('demo'):
                continue
            if split != BENCHMARKS[dataset][0]:
                continue
            match = re.search(r'(\d{8}_\d{6})\.json$', member.name)
            timestamp = match.group(1) if match else ''
            key = (dataset, split, model)
            if key not in selected or timestamp > selected[key][0]:
                selected[key] = (timestamp, member.name, data, hashlib.sha256(raw).hexdigest())
    issues, coverage, identities = [], [], {}
    total = 0
    target = output / 'records.jsonl.gz'
    temporary = output / 'records.jsonl.gz.tmp'
    with gzip.open(temporary, 'wt', encoding='utf-8') as stream:
        for dataset, (split, expected) in BENCHMARKS.items():
            for model in models:
                key = (dataset, split, model)
                if key not in selected:
                    issues.append({'dataset': dataset, 'model': model, 'error': 'missing result file'})
                    continue
                _, source, data, sha = selected[key]
                records = data.get('records', [])
                errors = Counter()
                nullable = Counter()
                index_counts = Counter(json.dumps(r.get('index'), sort_keys=True) for r in records)
                errors['duplicate_indices'] = sum(n - 1 for n in index_counts.values())
                if len(records) != expected:
                    errors['unexpected_record_count'] += 1
                if data.get('counts') != len(records):
                    errors['header_count_mismatch'] += 1
                identities[key] = {}
                for record in records:
                    for field in ('index', 'origin_query', 'prompt', 'prediction', 'ground_truth', 'score', 'prompt_tokens', 'completion_tokens', 'cost'):
                        if field not in record or record[field] is None:
                            if field in ('prediction', 'ground_truth') and field in record:
                                nullable['null_' + field] += 1
                            else:
                                errors['missing_' + field] += 1
                    for field in ('score', 'prompt_tokens', 'completion_tokens', 'cost'):
                        value = record.get(field)
                        if not finite_number(value) or value < 0 or (field == 'score' and value > 1):
                            errors['invalid_' + field] += 1
                    if not record.get('prompt'):
                        errors['empty_prompt'] += 1
                    identities[key][json.dumps(record.get('index'), sort_keys=True)] = digest(record.get('origin_query'))
                    row = dict(record, dataset_id=dataset, split=split, model_name=model,
                               record_index=record.get('index'), source_file=source)
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                failures = {k: v for k, v in errors.items() if v}
                entry = dict(dataset_id=dataset, split=split, model_name=model,
                             expected_records=expected, records=len(records),
                             source_file=source, source_sha256=sha, errors=failures, nullable_fields=dict(nullable))
                coverage.append(entry)
                if failures:
                    issues.append(entry)
                total += len(records)
    for dataset, (split, _) in BENCHMARKS.items():
        reference = identities.get((dataset, split, models[0]), {})
        for model in models[1:]:
            current = identities.get((dataset, split, model), {})
            missing = sorted(set(reference) - set(current))
            extra = sorted(set(current) - set(reference))
            different = [i for i in reference.keys() & current.keys() if reference[i] != current[i]]
            if missing or extra or different:
                issues.append(dict(dataset_id=dataset, model_name=model, error='query alignment mismatch',
                                   missing_indices=missing, extra_indices=extra, different_query_indices=different))
    temporary.replace(target)
    report = dict(source=SOURCE, source_config=SOURCE + '/blob/main/config/baseline_config_performance_cost.yaml',
                  archive=str(archive.resolve()), models=list(models),
                  model_aliases={'GPT-5 Medium': 'gpt-5'},
                  expected_records=sum(x[1] for x in BENCHMARKS.values()) * len(models),
                  loaded_records=total, expected_combinations=len(BENCHMARKS)*len(models),
                  loaded_combinations=len(coverage), complete=not issues, issues=issues, coverage=coverage)
    report['coverage_complete'] = len(coverage) == len(BENCHMARKS) * len(models) and all(
        r['records'] == r['expected_records'] and not r['errors'].get('duplicate_indices') for r in coverage)
    report['score_cost_complete'] = report['coverage_complete'] and not any(
        any(r['errors'].get(k) for k in ('missing_score', 'invalid_score', 'missing_cost', 'invalid_cost')) for r in coverage)
    (output / 'completeness.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('complete', 'loaded_records', 'expected_records', 'loaded_combinations', 'expected_combinations', 'issues')}, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--output', type=Path, default=Path('data/intern_flash_gpt5'))
    parser.add_argument('--models', nargs='+', default=MODELS)
    args = parser.parse_args()
    result = import_archive(args.archive, args.output, args.models)
    raise SystemExit(0 if result['complete'] else 1)
