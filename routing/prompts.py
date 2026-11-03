"""Shared router-visible contract. Never put solver identities here."""

CANDIDATES = ('Candidate A', 'Candidate B', 'Candidate C')
# Public-source rationale and identities live in README.md, outside router inputs.
DEFAULT_PROFILES = {
    'Candidate A': (
        'General and scientific reasoning capabilities; $0.18 input / $0.54 output per million tokens.'
    ),
    'Candidate B': (
        'General-purpose language, coding, and reasoning capabilities; $0.30 / $2.50.'
    ),
    'Candidate C': (
       'Coding and complex reasoning at medium effort; $1.25 / $10.00.'
    ),
}
SYSTEM_PROMPT = '''
You select a candidate language model to handle the current request.

You will receive:
- A list of candidates and descriptions of their capabilities and costs.
- The current request.

Assess the request's difficulty and the capabilities and costs described for each candidate. 
Select the candidate that best fits the request.

Return exactly one label:
Candidate A
Candidate B
Candidate C

Provide no explanation or additional text.
'''

PROMPT_OBJECTIVES = {
    'balanced': 'Balance cost and accuracy. Select the least expensive candidate that you expect to provide an accurate answer to the request.',
    'accuracy-first': 'Prioritize highest accuracy. Only downgrade to a cheaper candidate when you are confident it can accomplish the task.',
    'cost-first': 'Prioritize lowest cost. Only upgrade to a more expensive candidate when you do not believe a cheaper candidate can accomplish the task.',
}


def build_system_prompt(profiles=None, *, variant='default'):
    if variant != 'default' and variant not in PROMPT_OBJECTIVES:
        raise ValueError(f'Unknown system prompt variant: {variant}')
    profiles = DEFAULT_PROFILES if profiles is None else profiles
    if set(profiles) != set(CANDIDATES):
        raise ValueError('Profiles must describe exactly Candidate A, Candidate B, and Candidate C')
    prompt = SYSTEM_PROMPT
    if variant != 'default':
        prompt = prompt.replace('Select the candidate that best fits the request.', PROMPT_OBJECTIVES[variant])
    return prompt + '\n\n' + '\n'.join(
        f'{label}: {profiles[label]}'
        for label in CANDIDATES
    )
