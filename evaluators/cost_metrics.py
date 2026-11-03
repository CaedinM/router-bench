"""Add routing cost, total system cost, and score gap to a Phoenix experiment."""
import argparse
import json
import math
from pathlib import Path

from evaluators.downstream_metrics import PREFIXES, published_metrics
from routing.candidates import CANDIDATE_MODELS
from routing.pipeline import load_environment, make_phoenix_client

# Public USD rates per million tokens, checked October 27, 2026.
# https://docs.typesafe.ai/models
# https://developers.openai.com/api/docs/guides/decisions
# https://developers.openai.com/api/docs/pricing
PRICING_DATE = '2026-10-27'


def nonnegative_number(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f'Missing or invalid {field}')
    return value


def routing_cost(output, router_config):
    """Price recorded successful-response usage; never infer missing token counts."""
    if not isinstance(output, dict) or not isinstance(output.get('usage'), dict):
        raise ValueError('Router usage is missing or invalid')
    usage = output['usage']
    tokens = nonnegative_number(usage.get('input_tokens'), 'input_tokens')
    provider_model = (router_config.get('provider'), router_config.get('model'))
    pricing_date = PRICING_DATE
    if provider_model == ('jev', 'jev-1.13.0'):
        if router_config.get('base_url', 'https://api.typesafe.ai/v1').rstrip('/') != 'https://api.typesafe.ai/v1':
            raise ValueError('Pricing covers the standard TypeSafe endpoint only')
        cost = tokens * 0.042 / 1_000_000
        source = 'https://docs.typesafe.ai/models'
    elif provider_model == ('openai_decisions', 'gpt-6-luna'):
        base_url = router_config.get('base_url', 'https://api.openai.com/v1').rstrip('/')
        if base_url != 'https://api.openai.com/v1':
            raise ValueError('Pricing covers the standard global OpenAI endpoint only')
        details = usage.get('input_tokens_details')
        if not isinstance(details, dict):
            raise ValueError('OpenAI cache usage details are missing')
        cached = nonnegative_number(details.get('cached_tokens'), 'cached_tokens')
        writes = nonnegative_number(details.get('cache_write_tokens'), 'cache_write_tokens')
        if cached + writes > tokens:
            raise ValueError('Cache usage exceeds total input tokens')
        parameters = router_config.get('parameters', {})
        ttl = parameters.get('prompt_cache_options', {}).get('ttl', '30m')
        if ttl not in ('30m', '12h'):
            raise ValueError('Unknown cache write lifetime')
        write_rate = 0.125 if ttl == '30m' else 0.25
        multiplier = 2 if tokens > 272_000 else 1
        cost = ((tokens - cached - writes) * 0.10 + cached * 0.01 + writes * write_rate) * multiplier / 1_000_000
        source = 'https://developers.openai.com/api/docs/guides/decisions'
    elif provider_model == ('anthropic', 'claude-haiku-5-5'):
        if router_config.get('base_url', 'https://api.anthropic.com').rstrip('/') != 'https://api.anthropic.com':
            raise ValueError('Pricing covers the standard Anthropic endpoint only')
        if usage.get('inference_geo', 'global') != 'global' or usage.get('speed', 'standard') != 'standard':
            raise ValueError('Pricing covers standard global Anthropic inference only')
        output_tokens = nonnegative_number(usage.get('output_tokens'), 'output_tokens')
        cached = nonnegative_number(usage.get('cache_read_input_tokens'), 'cache_read_input_tokens')
        writes = nonnegative_number(usage.get('cache_creation_input_tokens'), 'cache_creation_input_tokens')
        # Anthropic reports uncached input separately from cache reads and writes.
        # All three count toward the 100k prompt-length pricing threshold.
        multiplier = 5 if tokens + cached + writes > 100_000 else 1
        write_cost = 0
        if writes:
            details = usage.get('cache_creation')
            if not isinstance(details, dict):
                raise ValueError('Anthropic cache write lifetimes are missing')
            short = nonnegative_number(details.get('ephemeral_5m_input_tokens'), 'ephemeral_5m_input_tokens')
            long = nonnegative_number(details.get('ephemeral_1h_input_tokens'), 'ephemeral_1h_input_tokens')
            if short + long != writes:
                raise ValueError('Anthropic cache write lifetimes do not match total writes')
            write_cost = short * 0.125 + long * 0.20
        cost = (tokens * 0.10 + cached * 0.01 + write_cost + output_tokens * 0.50) * multiplier / 1_000_000
        source = 'https://platform.claude.com/docs/en/models/haiku-5-5/overview'
        pricing_date = '2026-11-02'
    else:
        raise ValueError(f'No verified routing pricing for {provider_model}')
    return {'name': 'routing_cost', 'score': cost, 'metadata': {
        'currency': 'USD', 'pricing_date': pricing_date, 'pricing_source': source,
        'provider': provider_model[0], 'model': provider_model[1],
        'basis': 'recorded successful response usage',
    }}


def gap(output, expected):
    """Per-question oracle score minus selected score, among the active candidates."""
    if not isinstance(output, dict) or output.get('decision') not in CANDIDATE_MODELS:
        raise ValueError('A successful decision with a known candidate is required')
    model = CANDIDATE_MODELS[output['decision']]
    if output.get('selected_model') != model or not isinstance(expected, dict):
        raise ValueError('Selected model or reference output is invalid')
    scores = {}
    for candidate_model in CANDIDATE_MODELS.values():
        key = f'{PREFIXES[candidate_model]}_score'
        score = nonnegative_number(expected.get(key), key)
        if score > 1:
            raise ValueError(f'Reference score exceeds one: {key}')
        scores[candidate_model] = score
    best = max(scores.values())
    return {'name': 'gap', 'score': best - scores[model], 'metadata': {
        'selected_model': model, 'maximum_downstream_score': best,
        'downstream_score': scores[model], 'candidate_count': len(scores),
    }}


def make_cost_evaluators(router_config):
    def routing_cost_metric(output):
        return routing_cost(output, router_config)

    def total_system_cost(output, expected):
        router = routing_cost(output, router_config)
        downstream = next(metric for metric in published_metrics(output, expected)
                          if metric['name'] == 'downstream_cost')
        return {'name': 'total_system_cost', 'score': router['score'] + downstream['score'],
                'metadata': {**router['metadata'], **downstream['metadata'],
                             'routing_cost': router['score'], 'downstream_cost': downstream['score']}}

    return [routing_cost_metric, total_system_cost, gap]


def evaluate_experiment(client, experiment_id):
    experiment = client.experiments.get_experiment(experiment_id=experiment_id)
    # Use the run's saved configuration, never today's potentially changed config.
    router_config = experiment['experiment_metadata'].get('router_config', {})
    return client.experiments.evaluate_experiment(
        experiment=experiment, evaluators=make_cost_evaluators(router_config),
        print_summary=False, retries=0)


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
