# Router Bench

The purpose of this eval is to benchmark the cost-accuracy tradeoffs of using different models as model routers.

The pipeline generates routing decisions for over 12,416 prompts benchmark prompts and evluates them via data available here: https://github.com/ynulihao/LLMRouterBench

Our dataset was uploaded into Arize Phoenix to experiment across different models and prompts. Latency and token cost is tracked deterministally while evaluators are used to deterministically map outputs to downstream performance metrics.

## Evaluated Models
- Type Safe AI Jev 1.14
- OpenAI Decisions API
- Claude Haiku 5.5

## Routing Options
| Model | Role in benchmark | Why include it |
| --- | --- | --- |
| Intern-S1 | Value model | Cost-efficient lightweight model for straight-forward tasks |
| Gemini-2.5-flash | Mid-tier generalist | Stronger, higher-cost option, best for general intensive tasks. |
| GPT-5 Medium | Reasoning model | Highest cost, premium reasoning model. |

* Model names and benchmark data were not exposed to the routing model. Router only sees a brief description of model capability + token prices.

### Solver Model Baselines 
| Model | Mean Benchmark Accuracy | Mean Benchmark Cost /1000 prompts |
| --- | --- | --- |
| Intern-S1 | 44.93% | $3.79 |
| Gemini 2.5 Flash | 49.10% | $8.78 |
| GPT-5 Medium | 65.76% | $33.12 |

- Downstream accuracy is bounded by the capabilities of the models available for selection. Outputs for stronger models such as Claude Opus 5.5, GPT-6/6.1 and Gemini 4 were not present in the benchmark dataset thus were not served to the routers as options.

## Benchmarks
| Category | Benchmark | Prompts |
| --- | --- | --- |
| Mathematics | AIME | 60 |
| Mathematics | LiveMathBench | 121 |
| Coding | LiveCodeBench | 1,055 |
| Coding | SWE-Bench | 500 |
| Knowledge | GPQA | 198 |
| Knowledge | Humanity's Last Exam (HLE) | 2,158 |
| Knowledge | MMLU-Pro | 3,000 |
| Knowledge | SimpleQA | 4,326 |
| Instruction following | ArenaHard | 750 |
| Tool use | τ²-Bench | 278 |

## System Prompt Configurations
To observe and account for the impact of the system prompt on router decisions, decisions for each model were generated multiple times under different system prompt configurations.
- **Default:** Select the candidate that best fits the request.  
- **Balanced:** Explicitly instructed to balance cost and accuracy; choose the least expensive candidate expected to answer accurately.  
- **Accuracy-first:** Prioritize accuracy; choose a cheaper candidate only when confident it can accomplish the task.  
- **Cost-first:** Prioritize low cost; choose a more expensive candidate only when a cheaper one is not believed capable of accomplishing the task.  

## Results
### Accuracy & Cost
| Router | Prompt | Mean benchmark score | Total cost / 1,000 prompts |
|---|---|---:|---:|
| Oracle | -- | 73.92% | -- |
| Haiku | Accuracy-first | 56.15% | $19.45 |
| Jev | Accuracy-first | 53.30% | $15.94 |
| Haiku | Default | 51.94% | $15.95 |
| Haiku | Balanced | 51.14% | $15.45 |
| Jev | Default | 49.74% | $11.78 |
| Haiku | Cost-first | 49.33% | $10.97 |
| Jev | Cost-first | 47.69% | $9.28 |
| Jev | Balanced | 46.94% | $6.27 |
| OpenAI Decisions | Default | 46.67% | $4.26 |
| OpenAI Decisions | Accuracy-first | 45.04% | $4.13 |
| OpenAI Decisions | Cost-first | 44.93% | $3.92 |
| OpenAI Decisions | Balanced | 44.90% | $3.93 |

- Oracle represents the maximum possible accuracy achievable given our routing options.
- Total cost is the mean per-prompt routing plus published solver execution cost, multiplied by 1,000.

**Takeaway**: OpenAI Decisions was heavily biased towards selecting cheaper models in its routing and even selected the cheapest model (Intern-S1) 100% of the time under our cost-saving system prompt configuration.

### Mean latency (routing time only)
| Model | Mean Latency |
| --- | --- |
| Jev | 0.173 seconds |
| OpenAI Decisions | 0.312 seconds |
| Claude Haiku 5.5 | 1.911 seconds |

### Decisions
| Router | Prompt | Intern-S1 | Gemini 2.5 Flash | GPT-5 Medium |
|---|---|---:|---:|---:|
| Jev | Default | 52.1% | 37.8% | 10.1% |
| Jev | Balanced | 82.2% | 15.9% | 1.9% |
| Jev | Accuracy-first | 44.5% | 39.1% | 16.3% |
| Jev | Cost-first | 79.8% | 14.8% | 5.4% |
| OpenAI Decisions | Default | 87.9% | 12.0% | 0.1% |
| OpenAI Decisions | Balanced | 99.8% | 0.2% | <0.1% |
| OpenAI Decisions | Accuracy-first | 97.7% | 2.2% | 0.1% |
| OpenAI Decisions | Cost-first | 100.0% | 0.0% | 0.0% |
| Haiku | Default | 68.1% | 17.7% | 14.3% |
| Haiku | Balanced | 58.2% | 30.0% | 11.9% |
| Haiku | Accuracy-first | 46.9% | 28.8% | 24.3% |
| Haiku | Cost-first | 79.4% | 14.7% | 5.9% |

## Engineering Challenges

- **Avoiding repeated solver inference costs:** Running each selected solver on every benchmark prompt for every router configuration would be expensive. Instead, we uploaded a dataset containing the solvers' published outcomes and built Phoenix evaluators that match each routing decision to the selected solver's archived score and execution cost. This lets us compare configurations without spending on solver inference, while tracking routing cost and latency separately.
- **Normalizing different provider interfaces:** Jev, OpenAI Decisions, and Haiku have different request and response formats. The pipeline gives them a shared instruction and the same three choices, then normalizes their outputs to a candidate label. Haiku uses a constrained structured response; invalid decisions fail rather than silently falling back to a solver.
- **Accounting for different benchmark sizes:** Initially, averaging scores across individual prompts gave more weight to larger benchmarks, favoring routers that performed well on those datasets. We switched to calculating a mean score for each benchmark and then weighting the ten benchmark means equally, giving a clearer picture of performance across benchmarks.

## Takeaways
- How you word your system prompt matters. Jev accuracy increased by 6.36 points and mean cost more than doubled when subtly instructed to prioritize accuracy over balancing costs.
- Performance-Cost routing is a tradeoff. All of our configurations beat Intern-S1 baseline on accuracy while costing more, but lost to GPT-5 while costing less.

### Which Classifier Should I Use for Routing?
- OpenAI Decisions API when minimizing cost is the priority. Despite higher routing inference costs, Decisions favored cheaper models under all of our prompt configurations and lead to a cheaper total system cost than Jev at slightly lower downstream benchmark accuracy.
- Jev when prioritizing latency and accuracy. Jev had the most even distribution of routing decisions and achieved better mean benchmark accuracy than OpenAI Decisions across all configurations. It also responded 1.7x faster than OpenAI Decisions API and 11x faster than Haiku on average.
- Haiku when cost and latency are secondary, and accuracy is worth the premium. Haiku tends to pick stronger models leading to higher accuracy but also MUCH higher token costs.

## Limitations
- Routing only decides between a simple three-tiered model set and lacks more recent model families due to a lack of benchmark output data. A different model pool could change both the best routing decisions and the cost–score tradeoff. Therefore downstream results are not indicative of true performance in modern systems. They should only be used to consider the router’s relative performance against one another. With a larger token budget, I would have created output data for newer models such as 
- Our prompt configurations do not cover every possible routing strategy. Four instruction variants reveal sensitivity to wording and objectives, but they do not establish the best possible prompt or router.