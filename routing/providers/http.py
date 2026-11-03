"""Small HTTP transport shared by provider adapters; credentials stay in memory."""
import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen


class ProviderHTTPError(RuntimeError):
    def __init__(self, status_code, message, error_type=None):
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type


def post_json(url, payload, headers, timeout):
    request = Request(url, data=json.dumps(payload).encode(),
                      headers={'Content-Type': 'application/json', **headers}, method='POST')
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as exc:
        body = exc.read(16384).decode('utf-8', errors='replace').strip()
        # Do not expose credentials if a provider echoes request headers.
        for name, value in headers.items():
            if name.lower() in ('authorization', 'x-api-key') and value:
                body = body.replace(value, '[redacted]')
                if value.lower().startswith('bearer '):
                    body = body.replace(value[7:], '[redacted]')
        detail = f': {body}' if body else ''
        error_type = None
        try:
            parsed = json.loads(body)
            if isinstance(parsed, dict) and isinstance(parsed.get('detail'), dict):
                error_type = parsed['detail'].get('error_type')
        except (ValueError, TypeError):
            pass
        raise ProviderHTTPError(exc.code, f'Provider HTTP error {exc.code}{detail}', error_type) from None
