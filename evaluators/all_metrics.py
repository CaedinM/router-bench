"""Add all routing and downstream metrics to a Phoenix experiment."""
import argparse
import asyncio
import json
from pathlib import Path
from urllib.parse import quote

from evaluators.cost_metrics import make_cost_evaluators
from evaluators.downstream_metrics import published_metrics
from evaluators.router_metrics import router_metrics
from routing.pipeline import load_environment, make_phoenix_client


DEFAULT_CONCURRENCY = 64
TIMEOUT = 600


async def load_experiment(client, experiment_id):
    # The SDK's get_experiment also downloads a full JSON export with references.
    # Fetch only metadata and paginated task outputs here; load references once below.
    metadata = await client.experiments.get(experiment_id=experiment_id)
    runs, cursor = [], None
    while True:
        params = {'limit': 1000}
        if cursor:
            params['cursor'] = cursor
        response = await client._client.get(
            f'v1/experiments/{quote(experiment_id, safe="")}/runs', params=params, timeout=TIMEOUT)
        response.raise_for_status()
        page = response.json()
        runs.extend(page['data'])
        cursor = page.get('next_cursor')
        if not cursor:
            break
    return {**metadata, 'experiment_metadata': metadata.get('metadata', {}), 'task_runs': runs}


def evaluation_jobs(experiment, references, summary):
    """Compute existing metrics independently, without creating reference-heavy traces."""
    router_config = experiment['experiment_metadata'].get('router_config', {})
    cost, total, gap = make_cost_evaluators(router_config)
    for run in experiment['task_runs']:
        if run.get('error'):
            summary['skipped_failed_runs'] += 1
            continue
        output = run['output']
        expected = references.get(run['dataset_example_id'])
        summary['evaluated_runs'] += 1
        for evaluator, arguments in (
            (published_metrics, (output, expected)),
            (router_metrics, (output,)),
            (cost, (output,)),
            (total, (output, expected)),
            (gap, (output, expected)),
        ):
            try:
                result = evaluator(*arguments)
            except Exception as exc:
                summary['evaluation_errors'] += 1
                yield {'experiment_run_id': run['id'], 'name': evaluator.__name__,
                       'error': repr(exc)}
                continue
            for metric in result if isinstance(result, list) else [result]:
                yield {'experiment_run_id': run['id'], **metric}


async def evaluate_experiment(client, experiment_id, *, concurrency=DEFAULT_CONCURRENCY):
    """Use an AsyncClient to load references once and submit bounded concurrent writes."""
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
        raise ValueError('concurrency must be a positive integer')
    experiment = await load_experiment(client, experiment_id)
    dataset = await client.datasets.get_dataset(
        dataset=experiment['dataset_id'], version_id=experiment['dataset_version_id'],
        timeout=TIMEOUT)
    references = {example['id']: example['output'] for example in dataset}
    summary = {'evaluated_runs': 0, 'skipped_failed_runs': 0,
               'submitted_evaluations': 0, 'evaluation_errors': 0}
    # A shared iterator limits work to the worker count, without a task per metric.
    jobs = iter(evaluation_jobs(experiment, references, summary))

    async def submit():
        for job in jobs:
            await client.experiments.log_evaluation(**job, timeout=TIMEOUT)
            summary['submitted_evaluations'] += 1

    workers = [asyncio.create_task(submit()) for _ in range(concurrency)]
    try:
        await asyncio.gather(*workers)
    finally:
        # Submission errors propagate; stop outstanding writes before closing the client.
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
    return summary


async def run_evaluation(config, experiment_id, concurrency):
    client = make_phoenix_client(config, asynchronous=True)
    try:
        return await evaluate_experiment(client, experiment_id, concurrency=concurrency)
    finally:
        await client._client.aclose()


def positive_integer(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment_id', help='Phoenix experiment ID to evaluate')
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'routing/config.json')
    parser.add_argument('--concurrency', type=positive_integer, default=DEFAULT_CONCURRENCY,
                        help='Maximum concurrent metric submissions (default: 64)')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    load_environment()
    summary = asyncio.run(run_evaluation(config, args.experiment_id, args.concurrency))
    print(f"Submitted {summary['submitted_evaluations']} evaluations for {summary['evaluated_runs']} "
          f"runs; skipped {summary['skipped_failed_runs']} failed router tasks; "
          f"{summary['evaluation_errors']} evaluation errors.")
    if summary['evaluation_errors']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
