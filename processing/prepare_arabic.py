import json
from collections import Counter
from urllib.parse import quote

from datasets import load_dataset

from data_prep import ARABIC_DIR, ARABIC_PDF_DIR

ARCD_NAME = "hsseinmz/arcd"
ARCD_CONFIG = "plain_text"
NUM_ARTICLES = 8
WIKIPEDIA_URL = "https://ar.wikipedia.org/wiki/{title}"
QUESTION_MARK = "\u061f"
ARTICLES_FILE = "articles.json"
QUESTIONS_FILE = "questions.json"


def clean_question(text):
    """Strip list markers and any fragment that follows the final question mark."""
    text = text.strip().lstrip("-").strip()
    if QUESTION_MARK in text:
        text = text[: text.rindex(QUESTION_MARK) + 1]
    return text


def article_url(title):
    """Return the Arabic Wikipedia URL of an article title."""
    return WIKIPEDIA_URL.format(title=quote(title.replace(" ", "_")))


def load_rows():
    """Load every ARCD question from all splits."""
    dataset = load_dataset(ARCD_NAME, ARCD_CONFIG)
    return [row for split in dataset.values() for row in split]


def select_titles(rows, count):
    """Return the titles with the most questions, breaking ties alphabetically."""
    counts = Counter(row["title"] for row in rows)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [title for title, _ in ranked[:count]]


def build_records(rows, titles):
    """Build the article list and the evaluation questions for the selected titles."""
    ids = {title: f"ar{index:02d}" for index, title in enumerate(titles, start=1)}
    counts = Counter(row["title"] for row in rows)
    articles = [
        {
            "id": ids[title],
            "title": title,
            "url": article_url(title),
            "pdf": (ARABIC_PDF_DIR / f"{ids[title]}.pdf").as_posix(),
            "questions": counts[title],
        }
        for title in titles
    ]
    questions = [
        {
            "question_id": row["id"],
            "paper_id": ids[row["title"]],
            "text": clean_question(row["question"]),
            "gold_answers": [answer.strip() for answer in row["answers"]["text"]],
            "gold_evidence": [row["context"].strip()],
            "unanswerable": False,
        }
        for row in rows
        if row["title"] in ids
    ]
    return articles, questions


def write_json(path, payload):
    """Write a UTF-8 JSON file, creating parent directories when needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def main():
    """Select the Arabic Wikipedia articles to save as PDFs and store their ARCD questions."""
    rows = load_rows()
    articles, questions = build_records(rows, select_titles(rows, NUM_ARTICLES))
    write_json(ARABIC_DIR / ARTICLES_FILE, articles)
    write_json(ARABIC_DIR / QUESTIONS_FILE, questions)
    ARABIC_PDF_DIR.mkdir(parents=True, exist_ok=True)

    print(f"ARCD questions: {len(rows)}  selected articles: {len(articles)}  selected questions: {len(questions)}\n")
    for article in articles:
        print(f"{article['id']}  {article['questions']:>3} questions  {article['title']}")
        print(f"      {article['url']}")
    print(f"\nSave each article as a PDF named <id>.pdf in {ARABIC_PDF_DIR}/")


if __name__ == "__main__":
    main()