# Project purpose

This project compares different decision models and lightweight LLMs against each other as routers. It is a research and evaluation comparison, not an evaluation of a router for deployment. Interpret cost and performance analyses in that context.

# Router evaluation

Anonymize the candidate solving models presented to routers using consistent neutral identifiers. Do not expose model names or other identifying details to the decision model, so its choices are not influenced by potentially outdated or incorrect prior knowledge of those models. Describe capabilities truthfully using public documentation and published evaluations, with the same descriptions supplied to every routing approach. No separate training or calibration process is required. Do not derive descriptions from this project’s test outcomes. Treat the strongest available models as the top tier within this benchmark, not as claims about the current frontier. Preserve the original published execution costs for the comparison.

The decision model currently chooses among exactly three anonymous candidate labels: `Candidate A`, `Candidate B`, and `Candidate C`. Document a fixed mapping from these labels to the three active benchmark models and use it consistently across routing approaches and experiments. Keep this mapping outside router-visible inputs, and derive capability descriptions from public research and documentation, never from this project’s test outcomes. The active subset contains Intern-S1, Gemini 2.5 Flash, and GPT-5 Medium. The fixed mapping is Candidate A = intern-s1, Candidate B = gemini-2.5-flash, Candidate C = gpt-5. The sole local benchmark dataset is `data/intern_flash_gpt5/`; its saved development question IDs preserve the original split without the retired archives. Previous experiments remain historical.

# Preferences
Do not over-engineer. Do not add features or execute tasks that the user has not asked for. If you think a next step is a good idea, state it in your response. Do not execute it.