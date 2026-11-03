"""Copy published downstream metrics into an existing Phoenix experiment."""
import argparse
import json
import math
from pathlib import Path

from routing.candidates import CANDIDATE_MODELS
from routing.pipeline import load_environment, make_phoenix_client

PREFIXES = {'intern-s1': 'intern', 'gemini-2.5-flash': 'flash', 'gpt-5': 'gpt5'}
FIELDS = ('score', 'prompt_tokens', 'completion_tokens', 'cost')


def published_metrics(output, expected):
    """Return published metrics for the decision, without regrading or estimating costs."""
    if not isinstance(output, dict) or output.get('decision') not in CANDIDATE_MODELS:
        raise ValueError('A successful decision with a known candidate is required')
    model = CANDIDATE_MODELS[output['decision']]
    if output.get('selected_model') != model:
        raise ValueError('Selected model does not match the candidate mapping')
    if not isinstance(expected, dict):
        raise ValueError('Reference output is missing')
    prefix = PREFIXES[model]
    values = {}
    for field in FIELDS:
        key = f'{prefix}_{field}'
        value = expected.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'Missing or invalid reference field: {key}')
        values[field] = value
    if not 0 <= values['score'] <= 1 or values['cost'] < 0:
        raise ValueError('Reference score or cost is outside its valid range')
    # GPT-5's negative usage marks the entire paired question as unreliable for cost comparisons.
    for field in ('prompt_tokens', 'completion_tokens'):
        key = f'gpt5_{field}'
        value = expected.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'Missing or invalid reference field: {key}')
    invalid_usage = any(expected[f'gpt5_{field}'] < 0 for field in ('prompt_tokens', 'completion_tokens'))
    invalid_usage = invalid_usage or any(values[field] < 0 for field in ('prompt_tokens', 'completion_tokens'))
    metadata = {'selected_model': model, 'invalid_usage': invalid_usage}
    return [{'name': f'downstream_{field}', 'score': value, 'metadata': metadata}
            for field, value in values.items()]


def evaluate_experiment(client, experiment_id):
    experiment = client.experiments.get_experiment(experiment_id=experiment_id)
    return client.experiments.evaluate_experiment(
        experiment=experiment, evaluators=[published_metrics], print_summary=False, retries=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment_id', help='Phoenix experiment ID to evaluate')
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'routing/config.json')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    load_environment()
    evaluate_experiment(make_phoenix_client(config), args.experiment_id)
    print(f'Evaluation submitted for experiment {args.experiment_id}; inspect Phoenix for errors.')


if __name__ == '__main__':
    main()
