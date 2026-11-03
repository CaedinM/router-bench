# Haiku routing

This pipeline compares Claude Haiku 5.5 as a router against the decision-model
routers. It selects a candidate only; it does not call solving models.

From the project root:

```sh
.venv/bin/python -m pip install -r routing/llm_routing/requirements.txt
.venv/bin/python -m routing.llm_routing.pipeline
```

Set `ANTHROPIC_API_KEY` and `PHOENIX_BASE_URL` in the root `.env` and start
Phoenix as described in [the routing README](../README.md). Shell variables take
precedence. The default configuration uses the pinned existing dataset version
and the development split. Set `split` to `test` to use the test split. Use
`--config PATH` for a different configuration and `--output PATH` for a local
append-only decision log. Each invocation creates a new Phoenix experiment.

The system message is exactly `routing.prompts.build_system_prompt()` with the
configured variant (`default`, `balanced`, `accuracy-first`, or `cost-first`).
It lists Candidate A, Candidate B, and Candidate C with the same public-source
capability descriptions and historical token rates as the decision-model
pipeline. Only the example's `input.prompt` becomes the user message.

The fixed mapping remains Candidate A = `intern-s1`, Candidate B =
`gemini-2.5-flash`, Candidate C = `gpt-5` (Medium). These identities stay outside
router-visible inputs and are recorded in experiment metadata after routing.
Original published solver execution costs remain unchanged.

`RoutingDecision` constrains the `decision` field to those three labels and
forbids extra fields. The [Anthropic structured-output SDK](https://platform.claude.com/docs/en/build-with-claude/structured-outputs)
converts this Pydantic model into a decoding constraint through
`messages.parse(output_format=RoutingDecision)` and validates the response.
The response is JSON, for example `{"decision": "Candidate B"}`; the pipeline
extracts the label for the existing evaluation contract. Anthropic adds its own
format instructions for structured outputs; the shared system message is
unchanged. Refusals, truncated responses, and invalid decisions fail explicitly
without a fallback candidate. SDK retries are disabled; the existing Phoenix
pipeline retries failed tasks once and logs every attempt.

The [Claude API model ID](https://platform.claude.com/docs/en/models/haiku-5-5/overview)
is `claude-haiku-5-5`. Configuration supports `max_tokens` and `thinking` under
`parameters`; thinking otherwise uses the provider default. Router latency and
provider token usage are saved through the shared pipeline. All existing Phoenix
evaluators can consume the results, including routing and total system costs.
Haiku router pricing uses the public November 2, 2026 snapshot: $0.10 input /
$0.50 output per million tokens, with five times those rates for prompts over
100,000 input tokens (including cache reads and writes). Thinking tokens are
already included in output usage. Cache reads and writes are priced separately;
the estimate covers standard global inference and recorded successful responses.
Published solver execution costs remain unchanged.
