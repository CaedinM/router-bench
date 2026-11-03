"""Run one Phoenix experiment per router; persist decisions without evaluators."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from threading import Lock
from time import perf_counter
from uuid import uuid4

from routing.candidates import CANDIDATE_MODELS
from routing.prompts import build_system_prompt
from routing.router import ModelRouter

PROVIDERS = {
    'jev': ('routing.providers.jev', 'JevClassifier', 'TYPESAFE_API_KEY'),
    'openai_decisions': ('routing.providers.openai_decisions', 'OpenAIDecisionsClassifier', 'OPENAI_API_KEY'),
    'anthropic': ('routing.llm_routing.router', 'HaikuClassifier', 'ANTHROPIC_API_KEY'),
}


def load_environment():
    """Load the project .env without replacing variables already set by the caller."""
    from dotenv import load_dotenv

    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / '.env', override=False)


def build_router(router_config, system_prompt):
    """Instantiate one router with its configured provider classifier."""
    from importlib import import_module

    router_name = router_config['name']
    if not isinstance(router_name, str) or not router_name.strip():
        raise ValueError('Router name must be nonempty')
    if router_config['provider'] not in PROVIDERS:
        raise ValueError(f'Unknown provider: {router_config["provider"]}')

    provider_module_name, classifier_class_name, default_api_key_env = PROVIDERS[router_config['provider']]
    api_key_env = router_config.get('api_key_env', default_api_key_env)
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise ValueError(f'Missing API key environment variable: {api_key_env}')
    if not isinstance(router_config['model'], str) or not router_config['model'].strip():
        raise ValueError('Each router needs a model identifier')

    provider_options = {
        option_name: router_config[option_name]
        for option_name in ('base_url', 'parameters', 'timeout')
        if option_name in router_config
    }
    classifier = getattr(import_module(provider_module_name), classifier_class_name)(
        model=router_config['model'],
        system_prompt=system_prompt,
        api_key=api_key,
        **provider_options,
    )
    return ModelRouter(classifier)


def make_task(router, router_name, output_path, context):
    """Only the prompt crosses into the router; reference and metadata never do."""
    lock = Lock()

    def task(input, example):
        started = perf_counter()
        record = {**context, 'router': router_name, 'example_id': example['id'],
                  'timestamp': datetime.now(timezone.utc).isoformat()}
        try:
            record['decision'], record['usage'] = router.route(input['prompt'])
            record['selected_model'] = CANDIDATE_MODELS[record['decision']]
            record['status'] = 'ok'
        except Exception as exc:
            record.update(decision=None, selected_model=None, status='error', error_type=type(exc).__name__)
            raise
        finally:
            record['latency_seconds'] = perf_counter() - started
            with lock, Path(output_path).open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        return {'decision': record['decision'], 'selected_model': record['selected_model'],
                'latency_seconds': record['latency_seconds'], 'usage': record['usage']}

    return task


def select_router_config(config, router_name):
    """Select one router without initializing any other provider."""
    router_configs = [router_config for router_config in config.get('routers', []) if router_config['name'] == router_name]
    if len(router_configs) != 1:
        raise ValueError('Select exactly one configured router by its unique name')
    return router_configs[0]


def make_phoenix_client(config, *, asynchronous=False):
    """Configure local Phoenix storage before importing and creating its client."""
    configured_url = config.get('phoenix_url', 'PHOENIX_BASE_URL')
    if configured_url.startswith(('http://', 'https://')):
        base_url = configured_url
    else:
        base_url = os.environ.get(configured_url)
        if not base_url or not base_url.startswith(('http://', 'https://')):
            raise ValueError(f'{configured_url} must contain an HTTP(S) Phoenix URL')

    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault('PHOENIX_WORKING_DIR', str(root / 'phoenix/storage'))
    import httpx
    if asynchronous:
        from phoenix.client import AsyncClient
        client = AsyncClient(base_url=base_url)
    else:
        from phoenix.client import Client
        client = Client(base_url=base_url)
    # get_experiment has no timeout argument; it uses the underlying HTTP client.
    client._client.timeout = httpx.Timeout(600)
    return client


def load_dataset(client, config):
    """Load the selected split and validate prompts before creating an experiment."""
    split = config.get('split', 'development')
    if split not in ('development', 'test'):
        raise ValueError('split must be development or test')
    dataset = client.datasets.get_dataset(
        dataset=config.get('dataset', 'router-bench-intern-flash-gpt5-full-v1'),
        version_id=config.get('dataset_version'), splits=[split], timeout=600)
    if not len(dataset):
        raise ValueError('Selected dataset split is empty')
    for example in dataset:
        prompt = example['input'].get('prompt')
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f'Example {example["id"]} has no nonempty prompt')
    return dataset


def build_run_context(dataset, split, system_prompt):
    """Identify this run and the exact dataset version and instructions it uses."""
    return {
        'run_id': uuid4().hex,
        'dataset_id': dataset.id,
        'dataset_version': dataset.version_id,
        'split': split,
        'system_prompt_sha256': hashlib.sha256(system_prompt.encode()).hexdigest(),
    }


def run_router_experiment(client, dataset, router, router_config, system_prompt, output_path, context):
    """Save one router's decisions in Phoenix and the local decision log."""
    router_name = router_config['name']
    return client.experiments.run_experiment(
        dataset=dataset,
        task=make_task(router, router_name, output_path, context),
        experiment_name=f'{router_name}-{context.get("system_prompt_variant", "default")}-{context["run_id"]}',
        experiment_metadata={**context, 'router_config': router_config, 'system_prompt': system_prompt,
                             'candidate_models': CANDIDATE_MODELS},
        timeout=router_config.get('timeout', 60) + 10,
        retries=1,
    )


def run_pipeline(config, output_path, *, router_name, client=None):
    """Prepare the selected router and dataset, then run its decision experiment."""
    load_environment()
    variant = config.get('system_prompt_variant', 'default')
    system_prompt = build_system_prompt(config.get('candidate_profiles'), variant=variant)
    router_config = select_router_config(config, router_name)
    router = build_router(router_config, system_prompt)
    client = make_phoenix_client(config) if client is None else client
    dataset = load_dataset(client, config)
    context = build_run_context(dataset, config.get('split', 'development'), system_prompt)
    context['system_prompt_variant'] = variant
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    experiment = run_router_experiment(
        client, dataset, router, router_config, system_prompt, output_path, context)
    return {router_name: experiment}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config', type=Path,
        default=Path(__file__).resolve().with_name('config.json'),
        help='Configuration file (defaults to routing/config.json)',
    )
    parser.add_argument('--router', required=True, help='Run only this configured router name')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = args.output or Path('routing/results') / f'{uuid4().hex}.jsonl'
    run_pipeline(config, output, router_name=args.router)
    print(f'Decisions saved to {output.resolve()}')


if __name__ == '__main__':
    main()
