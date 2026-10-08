import json
from datetime import datetime
from pathlib import Path

FEEDBACK_DIR = Path("feedback")
FEEDBACK_FILE = "feedback.jsonl"
VALID_RATINGS = ("up", "down")


class FeedbackStore:
    """Append-only JSON Lines store for user ratings of generated answers."""

    def __init__(self, directory=FEEDBACK_DIR):
        """Point the store at a feedback file inside the given directory."""
        self.path = Path(directory) / FEEDBACK_FILE

    def add(self, question, answer, rating, sources, model, comment=""):
        """Append one rating record and return it."""
        if rating not in VALID_RATINGS:
            raise ValueError(f"rating must be one of {VALID_RATINGS}")
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "question": question,
            "answer": answer,
            "rating": rating,
            "comment": comment,
            "model": model,
            "sources": sources,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def load(self):
        """Return all saved feedback records."""
        if not self.path.exists():
            return []
        with open(self.path, encoding="utf-8") as file:
            return [json.loads(line) for line in file if line.strip()]

    def summary(self):
        """Return the total number of ratings and the thumbs up and down counts."""
        records = self.load()
        up = sum(1 for record in records if record["rating"] == "up")
        return {"total": len(records), "up": up, "down": len(records) - up}