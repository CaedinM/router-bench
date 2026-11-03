# Local Phoenix

Phoenix runs at http://127.0.0.1:6006. Its SQLite database and local state live in `phoenix/storage/` (ignored by Git). The server uses HTTP port 6006 and gRPC port 4317; Phoenix's gRPC listener binds all interfaces by default. Use a trusted network or firewall port 4317.

To install in the project virtual environment and start it:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r phoenix/requirements.txt
./phoenix/start.sh
```

Stop the server with Ctrl-C in its terminal. The database persists across restarts. For client code, set `PHOENIX_WORKING_DIR` to the absolute path of `phoenix/storage` and `PHOENIX_BASE_URL=http://127.0.0.1:6006`. The trace HTTP endpoint is `http://127.0.0.1:6006/v1/traces`.

The active dataset is `router-bench-intern-flash-gpt5-full-v1`: Intern-S1, Gemini 2.5 Flash, and GPT-5 Medium. It contains 30 development and 12,416 test questions. On a fresh Phoenix instance, run `.venv/bin/python phoenix/upload_dataset.py`. The script reads `data/intern_flash_gpt5/records.jsonl.gz` and the fixed question IDs in `data/intern_flash_gpt5/development_questions.json`; it refuses to append to an existing dataset.

Development contains three questions per benchmark. Membership preserves the original score-stratified smoke test and is never recalculated from active-pool scores. Keep test outcomes separate from prompt tuning. Previous use or duplicate questions across splits have not been audited.

The uploader also creates benchmark-specific splits within each partition, such as `development/aime` and `test/gpqa`. In the notebook, `upload_dataset()` adds missing benchmark splits to an existing dataset without appending examples or changing the development/test memberships. Set `benchmarks = None` for all benchmarks, or a list such as `["aime", "gpqa"]` to run only those benchmarks within the selected partition.

The routing prompt comes from the Gemini 2.5 Flash record. Original model fields remain in private reference outputs under `intern`, `flash`, and `gpt5` prefixes; metadata flags exact-text mismatches. Published costs are historical. Ten questions have invalid GPT-5 token counts; see `../data/README.md` for paired exclusions. Solver identities and original records must not be passed to a router. Pin the uploaded dataset version in each routing configuration before comparing runs.

## Routing experiments

Phoenix experiments are the recommended framework for comparing routers, with routing logic implemented in Python. Run one experiment per router configuration (decision model or routing method, prompt where applicable, public-source candidate descriptions, and routing objective), using the same dataset version and evaluation examples for each comparison.

The experiment task receives the question prompt and anonymous candidate descriptions based on public documentation and published evaluations, and returns exactly one of `Candidate A`, `Candidate B`, or `Candidate C`. Document a fixed mapping to the active models and reuse it across experiments, keeping the mapping and reference records outside router-visible inputs. The development split is a smoke test. No separate calibration process is required; keep prompt tuning separate from test outcomes.

After selection, evaluators use the private mapping to look up the selected model's published score and execution cost in the reference fields. This evaluates routing decisions without rerunning the solving models. Preserve the original historical solving costs and apply the documented invalid-usage exclusions consistently across candidates. Record the router's own measured cost and latency separately from those historical solving costs. These experiments compare routing approaches for research; they do not establish deployment performance.

The decision-only pipeline is implemented in `routing/pipeline.py`; see [routing setup and independent per-model commands](../routing/README.md). Jev and OpenAI Decisions have separate provider adapters. Each invocation runs only the explicitly selected routing model and saves decisions without evaluators.
