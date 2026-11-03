"""Copy recorded router latency and token usage into a Phoenix experiment."""
import argparse
import json
import math
from pathlib import Path

from routing.pipeline import load_environment, make_phoenix_client


def router_metrics(output):
    """Copy available numeric metrics from a successful router output."""
    if not isinstance(output, dict):
        raise ValueError('Router output is missing')
    latency = output.get('latency_seconds')
    if isinstance(latency, bool) or not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency < 0:
        raise ValueError('Router latency is missing or invalid')
    results = [{'name': 'router_latency_seconds', 'score': latency}]
    usage = output.get('usage')
    if usage is None:
        return results
    if not isinstance(usage, dict):
        raise ValueError('Router usage must be an object or null')
    for field in ('input_tokens', 'output_tokens'):
        value = usage.get(field)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'Router {field} is invalid')
        results.append({'name': f'router_{field}', 'score': value})
    return results


def evaluate_experiment(client, experiment_id):
    experiment = client.experiments.get_experiment(experiment_id=experiment_id)
    return client.experiments.evaluate_experiment(
        experiment=experiment, evaluators=[router_metrics], print_summary=False, retries=0)


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
