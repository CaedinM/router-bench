"""TypeSafe Jev System One choice adapter."""
from routing.prompts import CANDIDATES
from routing.providers.http import ProviderHTTPError, post_json


class JevClassifier:
    def __init__(self, *, model, system_prompt, api_key,
                 base_url='https://api.typesafe.ai/v1', parameters=None, timeout=60):
        self.model = model
        self.system_prompt = system_prompt
        self.api_key = api_key
        self.base_url = base_url
        self.parameters = dict(parameters or {})
        self.timeout = timeout
        if {'model', 'state', 'questions'} & self.parameters.keys():
            raise ValueError('parameters cannot override model, state, or questions')

    def predict(self, prompt):
        return self.predict_with_usage(prompt)[0]

    def predict_with_usage(self, prompt):
        url = self.base_url.rstrip('/') + '/systemone'
        payload = {**self.parameters, 'model': self.model, 'state': prompt, 'questions': {
            'route': {'type': 'choice', 'instructions': self.system_prompt,
                      'criteria': {label: label for label in CANDIDATES}},
        }}
        headers = {'Authorization': f'Bearer {self.api_key}'}
        try:
            response = post_json(url, payload, headers, self.timeout)
        except ProviderHTTPError as exc:
            if exc.status_code != 400 or exc.error_type != 'max_tokens_exceeded':
                raise
            response = post_json(url, payload,
                                 {**headers, 'TypeSafe-Danger-Zone-Long-Context': 'true'},
                                 self.timeout)
        return response['answers']['route']['choice'], response.get('usage')
