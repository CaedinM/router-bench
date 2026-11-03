"""Load the fixed 30-question Phoenix development split."""
import json
from pathlib import Path

DEVELOPMENT_PATH = Path(__file__).resolve().parents[1] / 'data/intern_flash_gpt5/development_questions.json'


def load_development(path=DEVELOPMENT_PATH):
    questions = json.loads(Path(path).read_text())['questions']
    selected = {(q['dataset_id'], q['split'], q['record_index']) for q in questions}
    if len(questions) != 30 or len(selected) != 30:
        raise ValueError('Expected 30 distinct development questions')
    return selected
