"""Read the imported benchmark records, preserving published scores and costs."""
import gzip
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / 'data' / 'intern_flash_gpt5'

# Exclude the entire query across models so comparisons retain paired samples.
INVALID_USAGE_QUERIES = frozenset(
    [('hle', 'test', i) for i in (554, 1144, 1265, 2000, 2072)]
    + [('simpleqa', 'test', i) for i in (1137, 2574, 2683, 3592, 4113)]
)


def completeness(data_dir=DATA_DIR):
    return json.loads((Path(data_dir) / 'completeness.json').read_text())


def iter_records(data_dir=DATA_DIR, *, dataset=None, model=None, require_complete=True,
                 exclude_invalid_usage=True):
    """Yield original record fields plus dataset/split/model and provenance.

    Missing scores and costs are never replaced with zeros. By default reject
    an import that failed coverage, field validation, or query alignment.
    By default, remove all models for queries with invalid published usage.
    Set exclude_invalid_usage=False to inspect the original records.
    """
    report = completeness(data_dir)
    if require_complete and not report['complete']:
        raise ValueError('Incomplete benchmark data; inspect completeness.json before use.')
    with gzip.open(Path(data_dir) / 'records.jsonl.gz', 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if exclude_invalid_usage and (row['dataset_id'], row['split'], row['record_index']) in INVALID_USAGE_QUERIES:
                continue
            if dataset is not None and row['dataset_id'] != dataset:
                continue
            if model is not None and row['model_name'] != model:
                continue
            yield row


def load_dataframe(data_dir=DATA_DIR, **filters):
    """Optional pandas interface; pandas must be installed by the caller."""
    import pandas as pd
    return pd.DataFrame.from_records(iter_records(data_dir, **filters))
