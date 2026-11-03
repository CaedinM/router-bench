# Routing decisions

For the Claude Haiku 5.5 LLM router with Pydantic-constrained decisions, see
[`llm_routing/README.md`](llm_routing/README.md).

Run each routing model independently against a Phoenix dataset split. Each invocation creates one router object and one Phoenix experiment, and appends one JSONL record per attempted example. Adding a router to the configuration does not execute existing routers: `--router` is required and selects only that entry. An explicit rerun creates a new experiment; this pipeline does not automatically resume or deduplicate repeated invocations.

Every run automatically uses up to 16 concurrent routing tasks, fixed by `ROUTING_CONCURRENCY` in `pipeline.py`, including Haiku runs. Phoenix's async runner dispatches the blocking provider calls to worker threads; JSONL writes remain locked and records may arrive in completion order. The concurrency setting is saved in experiment metadata and each local record. Provider rate limits and Phoenix write throughput can limit speedup, and concurrent load can affect measured router latency. Existing commands are unchanged.

## Run

Install the existing Phoenix dependencies with `.venv/bin/python -m pip install -r phoenix/requirements.txt` and start Phoenix with `./phoenix/start.sh`. The adapters use Python's standard library and require no additional packages.

Put `TYPESAFE_API_KEY` for Jev and `OPENAI_API_KEY` for OpenAI Decisions in the project's root `.env` file. The pipeline loads this file automatically, including when called from Python; existing shell environment variables take precedence. Only the selected router's key is required. Model IDs, endpoints, provider parameters, timeouts, and dataset selection live in `config.json`; edit it to customize the run. The supplied model IDs are `jev-1.13.0` and `gpt-6-luna`; access depends on your provider account.

From the project root, run either pipeline independently:

Set `PHOENIX_BASE_URL=http://127.0.0.1:6006` (or your Phoenix server URL) in `.env`. The configuration's `"phoenix_url": "PHOENIX_BASE_URL"` resolves that environment variable after `.env` is loaded. A literal HTTP(S) URL is also supported in the configuration. Missing or invalid environment URLs fail before client creation.

```sh
.venv/bin/python -m routing.pipeline --router jev
.venv/bin/python -m routing.pipeline --router openai-decisions
```

Development is the default (30 smoke-test questions). To run test questions, explicitly set `split` to `test` in your configuration. Set `dataset_version` to a Phoenix version ID to pin separate runs to the same immutable dataset version. Without it, each invocation loads the latest version; compare the saved version IDs before comparing runs.

Results appear in Phoenix and in a unique file under `routing/results/`. Use `--output /absolute/path/decisions.jsonl` to choose the append-only local log. Records contain run ID, router name, example ID, dataset/version/split, shared prompt hash, decision, status, measured latency, and the provider's usage fields. Phoenix retries a failed task once; the local log records each attempt, so a failed record may be followed by a successful one for the same example. Successful Phoenix outputs contain the decision, latency, and usage; usage is null when the provider does not supply it. Usage from a successful Jev long-context retry describes that response, not necessarily any usage or billing for the rejected request. Local error logs record exception type; Phoenix retains exception details, including provider error responses. Completed local records remain available if an invocation is interrupted.

## Live end-to-end test

With Phoenix running and the selected provider's API key in `.env`, run:

```sh
RUN_ROUTING_E2E=1 .venv/bin/python -m unittest discover -s tests -p 'test_routing_e2e.py' -v
```

This creates a real Jev experiment with the pipeline's fixed concurrency of 16 and makes paid provider requests. It forces the Phoenix `development` split, checks that it contains exactly 30 distinct examples before routing, and pins that version for the run. It then reads the experiment back from Phoenix and checks all 30 decisions, candidate mappings, usage, and latencies against the local JSONL log. A failed final task fails the test; successful retries are allowed. Experiments and logs under `routing/results/e2e-*.jsonl` remain available for inspection, and the test prints their identifiers.

Set `ROUTING_E2E_ROUTER=openai-decisions` to test that configured router instead. To test Haiku, set both `ROUTING_E2E_CONFIG=routing/llm_routing/config.json` and `ROUTING_E2E_ROUTER=haiku`. The test is skipped in normal test discovery unless `RUN_ROUTING_E2E=1` is explicitly set. It tests decision generation and persistence; run evaluators separately when needed.

## Prompt objective experiments

Three additional variants share the default contract and public-source candidate profiles:

- `balanced`: balance cost and accuracy; select the least expensive candidate expected to answer accurately.
- `accuracy-first`: prioritize highest accuracy; downgrade only when confident the cheaper candidate can accomplish the task.
- `cost-first`: prioritize lowest cost; upgrade only when a cheaper candidate is not believed capable of accomplishing the task.

Set `system_prompt_variant` in `routing/config.json` to `default`, `balanced`, `accuracy-first`, or `cost-first`. The supplied configuration selects `default`. For example, select the accuracy-first prompt with:

```json
"system_prompt_variant": "accuracy-first"
```

Then run routing as usual, once for each router you want to compare:

```sh
.venv/bin/python -m routing.pipeline --router jev
.venv/bin/python -m routing.pipeline --router openai-decisions
```

Each command runs only the selected router with the configured prompt. The variant name, full prompt, and prompt hash are saved with each experiment. Collect evaluations afterward using the combined metrics command below.

## Shared routing contract

`ModelRouter` retains the classifier `predict(prompt)` interface. Each provider has its own file under `providers/`; register new classifier adapters in `pipeline.PROVIDERS`. The original explicit classifier-to-model mapping remains supported for other callers.

All routers receive exactly the same instruction string from `prompts.py`, the same question prompt, and the options `Candidate A`, `Candidate B`, and `Candidate C` in that order. Jev receives the instructions in its choice question, since its API has no chat system role; OpenAI Decisions receives that same string in its choice question. Only `input.prompt` from each Phoenix example is forwarded. Dataset reference outputs, extra input fields, metadata, and solver identities are excluded. Provider parameters cannot override the prompt or choices. Unknown labels fail validation.

The active mapping is Candidate A = Intern-S1 (`intern-s1`), Candidate B = Gemini 2.5 Flash (`gemini-2.5-flash`), Candidate C = GPT-5 Medium (`gpt-5`). This mapping stays outside router inputs and is saved in new experiment metadata. Previous runs used DeepSeek / Claude / GPT-5; do not re-evaluate those historical runs using the new mapping.

`prompts.py` supplies the current short system prompt and anonymous capability descriptions. Both routers receive exactly the same text. The longer prompt in `proposed_pool_prompts.py` is a proposal and is not active. The descriptions are supported by the [Intern-S1 model card](https://huggingface.co/internlm/Intern-S1), [Gemini 2.5 Flash documentation](https://ai.google.dev/gemini-api/docs/models/gemini-2.5-flash), and [GPT-5 documentation](https://developers.openai.com/api/docs/models/gpt-5). No local test outcomes were used to describe capabilities. Historical input/output USD rates per million tokens in the prompt are A = 0.18/0.54, B = 0.30/2.50, C = 1.25/10.00. Preserve each archived execution cost; do not recompute it from these rates.

The active dataset is `router-bench-intern-flash-gpt5-full-v1`, imported from the original release to `data/intern_flash_gpt5/`. It retains the previous 30 development question memberships and 12,416 test questions; the shared routing prompt comes from the Gemini 2.5 Flash record. Both runs must pin the same dataset version. Only question prompts are sent to providers.

This layer selects candidates only: it does not invoke solvers, compute scores or published execution costs, or run evaluators. Historical reference fields remain untouched. Provider-specific API contracts are documented in the [Jev API reference](https://docs.typesafe.ai/api/system-one) and [OpenAI Decisions reference](https://developers.openai.com/api/reference/resources/decisions/methods/create).

## Collect all metrics

After a routing experiment completes, run all evaluators with one command:

```sh
.venv/bin/python -m evaluators.all_metrics EXPERIMENT_ID
# Optional: adjust the number of concurrent metric writes (default: 64).
.venv/bin/python -m evaluators.all_metrics EXPERIMENT_ID --concurrency 32
```

Find the experiment ID in Phoenix. The command uses Phoenix's async client, loads task outputs through pagination, and fetches the experiment's pinned reference dataset once with a 600-second timeout. It computes the existing metrics locally and submits individual values through the evaluation API, with at most 64 concurrent writes by default. It does not create evaluation traces containing full prompts and solver records. Database write throughput still limits completion time.

The evaluators remain independent: missing cost inputs produce error evaluations without preventing downstream scores, router metrics, or gap values when their own inputs are valid. Original costs, pricing metadata, and invalid-usage flags are preserved. Failed router tasks are skipped. The command prints submission/error counts and exits nonzero for evaluator or submission errors; a submission failure cancels outstanding work and leaves already saved values intact. Running it again upserts evaluations with the same run IDs and metric names. Omitted optional usage metrics remain missing.

Python callers can use `await evaluators.all_metrics.evaluate_experiment(async_client, experiment_id, concurrency=64)`. The caller owns that client's lifecycle; the CLI closes its client automatically.

### Downstream metrics

For each successful decision, the command records `downstream_score`, `downstream_prompt_tokens`, `downstream_completion_tokens`, and `downstream_cost` as separate Phoenix evaluations, using the experiment's dataset version and its reference output. Cost is the original published execution cost, not an estimate. Scores are preserved as published; some are 0.5. No solver is called and no response is regraded. Router usage and latency remain in the task output separately.

Each metric has `selected_model` and `invalid_usage` metadata. The latter flags a question for every candidate when GPT-5 has negative token usage; future comparisons should exclude all candidates for those paired questions, as described in the data README. Values remain intact for inspection. Failed router tasks or missing/invalid reference fields do not receive invented scores; inspect Phoenix for evaluation errors. The command does not compute summary statistics.

### Router metrics

The command records `router_latency_seconds`, `router_input_tokens`, and `router_output_tokens` as Phoenix evaluations. Missing usage or token fields remain missing, rather than becoming zero. The original task output remains unchanged. Latency and usage describe the router, separate from the selected solver's downstream metrics. Jev usage after a long-context retry may omit usage for the rejected request. Failed tasks and invalid recorded values do not receive invented metrics; inspect Phoenix for evaluation errors.

### Cost and gap metrics

The command adds three Phoenix evaluation columns without calling either decision or solving models:

- `routing_cost`: estimated USD cost of the recorded routing API usage for that question.
- `total_system_cost`: routing cost plus the selected solver's original published `downstream_cost`, in USD. Published costs are never repriced.
- `gap`: maximum published downstream score across Candidate A, B, and C for that question, minus the selected candidate's score. Lower is better; zero means an optimal score, including ties. Scores from models outside this active pool are excluded. This uses the actual available maximum, so if all candidates score zero, the gap is zero. Partial scores are preserved.

Pricing is a fixed public-rate snapshot checked October 27, 2026. [TypeSafe's model reference](https://docs.typesafe.ai/models) lists Jev `jev-1.13.0` at $0.042 per million input tokens with free output. The [OpenAI Decisions guide](https://developers.openai.com/api/docs/guides/decisions) lists `gpt-6-luna` at $0.10 per million uncached input tokens with free output; [standard cache pricing](https://developers.openai.com/api/docs/pricing) is $0.01 for reads, $0.125 for 30-minute writes, and $0.25 for 12-hour writes per million tokens. Cache reads and writes are already included in total input tokens and are priced separately without double counting. The request's saved `prompt_cache_options.ttl` selects the write rate (default `30m`). The [model documentation](https://developers.openai.com/api/docs/models/gpt-6-luna) doubles input and cache rates above 272,000 input tokens.

The evaluator uses the experiment's saved `router_config`, not the current routing model configuration. Pricing metadata includes currency, snapshot date, model, provider, and source. These estimates cover standard public pricing at the default global endpoints, without account credits, negotiated discounts, or unrecorded failed/retried requests. Unknown models, missing input usage, and missing OpenAI cache details produce evaluation errors instead of fabricated zero costs. The gap evaluation is independent and remains available without router usage. Total system cost retains `selected_model` and `invalid_usage` metadata from downstream evaluation; exclude flagged paired questions consistently from cost comparisons, while preserving their recorded values for inspection.

Each row is a per-question cost. Sum `routing_cost` or `total_system_cost` across the same eligible questions to obtain experiment totals; Phoenix's mean evaluation score is a per-question average, not a total bill. Missing cost evaluations make a sum incomplete.

Claude Haiku 5.5 router costs use the November 2, 2026 public-rate snapshot,
including output/thinking usage, cache pricing, and the higher rates above
100,000 total input tokens. See [the Haiku pipeline notes](llm_routing/README.md)
for the rates and scope. Archived solver execution costs are preserved.
