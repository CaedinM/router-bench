"""A router delegates provider setup to its classifier's predict method."""
from routing.prompts import CANDIDATES


class ModelRouter:
    def __init__(self, classifier, models=None):
        self.classifier = classifier
        # Explicit legacy mappings remain supported for non-LLM classifiers.
        self.models = dict(models) if models is not None else {label: label for label in CANDIDATES}

    def route(self, prompt: str):
        """Return the validated decision and provider usage, when available."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError('prompt must be a nonempty string')
        if callable(getattr(type(self.classifier), 'predict_with_usage', None)):
            prediction, usage = self.classifier.predict_with_usage(prompt)
        else:
            prediction, usage = self.classifier.predict(prompt), None
        if isinstance(prediction, str):
            prediction = prediction.strip()
        if prediction not in self.models:
            raise ValueError(f'Invalid routing decision: {prediction!r}')
        return self.models[prediction], usage
