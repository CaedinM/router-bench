# Active benchmark data

Source: https://github.com/ynulihao/LLMRouterBench

The sole local benchmark dataset is `intern_flash_gpt5/`. It contains the three active candidates across ten published benchmarks:

- Candidate A = Intern-S1 (`intern-s1`)
- Candidate B = Gemini 2.5 Flash (`gemini-2.5-flash`)
- Candidate C = GPT-5 Medium (`gpt-5`)

This mapping stays outside router-visible inputs. Candidate descriptions come from public documentation and published evaluations, never local test outcomes.

## Files

- `intern_flash_gpt5/records.jsonl.gz`: 37,338 original records across 12,446 questions and 30 benchmark/model combinations. Original fields, model identifiers, responses, scores, token counts, and published execution costs are preserved. Added fields identify the dataset, split, model, record index, and source file. Model outputs are data and are never executed.
- `intern_flash_gpt5/completeness.json`: source filenames and SHA-256 hashes, coverage checks, and source-quality caveats. Coverage and score/cost completeness pass; strict completeness does not, for the issues below.
- `intern_flash_gpt5/development_questions.json`: the fixed 30 development question keys (`dataset_id`, `split`, `record_index`), three per benchmark. These preserve the prior score-stratified selection without recalculating membership from active-pool outcomes. The source hash identifies the retired three-model archive used for that selection. All remaining 12,416 questions belong to test.

The earlier five-model archive and historical three-model copy have been removed. Existing Phoenix experiments remain historical; do not reinterpret their candidate labels using the active mapping.

## Source caveats

Ten GPT-5 records have negative completion-token counts: HLE indices 554, 1144, 1265, 2000, 2072; SimpleQA indices 1137, 2574, 2683, 3592, 4113. Their published costs are zero and potentially unreliable. The source also has a query-text mismatch for GPT-5 at HLE index 1494. These original values remain unchanged.

The reader excludes all three candidates for each of the ten invalid-usage questions by default, leaving 37,308 records across 12,436 paired questions. Pass `exclude_invalid_usage=False` to inspect every original row. Code reading gzip directly must apply the same paired exclusions for comparisons. Published costs describe the original benchmark executions and must not be recomputed from current API rates. Null predictions or ground truths in the release remain intact; see the report's `nullable_fields` entries.

## Rebuild

From the project root, using the original release archive (Python standard library only):

```sh
python3 scripts/load_benchmarks.py /path/to/bench-release.tar.gz
```

Defaults select Intern-S1, Gemini 2.5 Flash, and GPT-5 and write to `data/intern_flash_gpt5/`. The command retains `development_questions.json`. It exits nonzero for the documented source caveats even with complete coverage; review the report before upload.

## Load

```python
from routing.benchmark_data import completeness, iter_records

report = completeness()
assert report['coverage_complete'] and report['score_cost_complete']
# Explicitly accept the documented source caveats after reviewing the report.
records = iter_records(require_complete=False)
```

The reader defaults to the active dataset and rejects strict source-quality failures unless explicitly accepted. Optional pandas support is available through `routing.benchmark_data.load_dataframe(require_complete=False)`.

## Phoenix upload

On a fresh Phoenix instance, run `.venv/bin/python phoenix/upload_dataset.py`. It loads the saved development IDs and active records, creates `router-bench-intern-flash-gpt5-full-v1`, and refuses to append to an existing dataset. Pin the returned dataset version in each routing configuration before comparing runs. Only the Gemini 2.5 Flash question prompt is forwarded to routers; original solver identities and records stay in private reference outputs.
