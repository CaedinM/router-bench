# Router Bench

A research benchmark comparing decision models and lightweight LLMs as model routers.

Read the [evaluation write-up](blog/router-bench-cost-accuracy-of-llm-routers.html) for the motivation, notebook walkthrough, engineering decisions, results, and takeaways. Download the HTML file and open it in a browser to view the formatted article.

## Repository structure

```text
blog/                 HTML evaluation write-up and visualizations
notebooks/            Notebook for running and summarizing routing experiments
routing/              Routing pipeline, provider adapters, prompts, and configuration
  providers/          Jev and OpenAI Decisions adapters and HTTP utilities
  llm_routing/        Claude Haiku router and configuration
  candidates.py       Private mapping from anonymous labels to solving models
  prompts.py          Shared routing instructions and anonymous candidate profiles
evaluators/           Published downstream metrics, router metrics, cost, and score gap
data/                 Active benchmark records, fixed development IDs, and provenance
phoenix/              Local Phoenix startup and dataset-upload entry points
scripts/              Benchmark source import utilities
tests/                Automated checks and opt-in live integration tests
.env.example          Environment configuration template
pyproject.toml        Python package configuration and dependencies
```

## Run experiments

Use Python 3.12 or newer. From the repository root:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -e ".[notebook]"
cp .env.example .env
.venv/bin/python -m ipykernel install --user --name router-bench --display-name "Router Bench"
.venv/bin/python -m jupyterlab notebooks/run_routing_experiments.ipynb
```

Fill in `.env` with the API keys for the routers you plan to run.

Open the [routing experiment notebook](notebooks/run_routing_experiments.ipynb), select the **Router Bench** kernel, and follow its cells. It defaults to the 30-question development split. Routing requests incur API costs; solver outcomes are read from the local benchmark records.

## Component documentation

- [Routing pipeline and CLI](routing/README.md)
- [Claude Haiku routing](routing/llm_routing/README.md)
- [Benchmark data and provenance](data/README.md)
- [Phoenix setup](phoenix/README.md)
