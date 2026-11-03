"""OpenAI Decisions adapter; uses native choice questions, not chat generation."""
from routing.prompts import CANDIDATES
from routing.providers.http import post_json


class OpenAIDecisionsClassifier:
    def __init__(self, *, model, system_prompt, api_key,
                 base_url='https://api.openai.com/v1', parameters=None, timeout=60):
        self.model = model
        self.system_prompt = system_prompt
        self.api_key = api_key
        self.base_url = base_url
        self.parameters = dict(parameters or {})
        self.timeout = timeout
        if {'model', 'input', 'questions'} & self.parameters.keys():
            raise ValueError('parameters cannot override model, input, or questions')

    def predict(self, prompt):
        return self.predict_with_usage(prompt)[0]

    def predict_with_usage(self, prompt):
        response = post_json(
            self.base_url.rstrip('/') + '/decisions',
            {**self.parameters, 'model': self.model, 'input': prompt, 'questions': [
                {'type': 'choice', 'name': 'route', 'instructions': self.system_prompt,
                 'choices': [{'value': label} for label in CANDIDATES]},
            ]}, {'Authorization': f'Bearer {self.api_key}'}, self.timeout,
        )
        answers = response['answers']
        if len(answers) != 1 or answers[0].get('type') != 'choice' or answers[0].get('name') != 'route':
            raise ValueError('Provider returned an unexpected answer')
        return answers[0]['choice'], response.get('usage')
