"""Claude Haiku router using native structured outputs and Pydantic validation."""
from typing import Literal

from anthropic import Anthropic
from pydantic import BaseModel, ConfigDict


class RoutingDecision(BaseModel):
    model_config = ConfigDict(extra='forbid')

    decision: Literal['Candidate A', 'Candidate B', 'Candidate C']


class HaikuClassifier:
    def __init__(self, *, model, system_prompt, api_key,
                 base_url='https://api.anthropic.com', parameters=None, timeout=60):
        self.model = model
        self.system_prompt = system_prompt
        self.parameters = dict(parameters or {})
        # Only expose generation settings; request/schema overrides could leak data
        # or bypass the anonymous routing contract.
        if self.parameters.keys() - {'max_tokens', 'thinking'}:
            raise ValueError('parameters only supports max_tokens and thinking')
        self.client = Anthropic(api_key=api_key, base_url=base_url,
                                timeout=timeout, max_retries=0)

    def predict(self, prompt):
        return self.predict_with_usage(prompt)[0]

    def predict_with_usage(self, prompt):
        response = self.client.messages.parse(
            **{'max_tokens': 4096, **self.parameters},
            model=self.model,
            system=self.system_prompt,
            messages=[{'role': 'user', 'content': prompt}],
            output_format=RoutingDecision,
        )
        if response.stop_reason != 'end_turn' or response.parsed_output is None:
            raise ValueError(f'Router did not return a complete decision: {response.stop_reason}')
        decision = RoutingDecision.model_validate(response.parsed_output)
        usage = response.usage.model_dump(mode='json') if response.usage is not None else None
        return decision.decision, usage
