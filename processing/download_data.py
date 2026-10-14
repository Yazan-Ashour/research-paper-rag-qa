import hashlib
import json
import time
from datetime import datetime

import requests
from datasets import load_dataset

from data_prep import (
    DATA_DIR,
    DATASET_NAME,
    DATASET_REVISION,
    MIN_QUESTIONS,
    NUM_PAPERS,
    PAPERS_FILE,
    PDF_DIR,
    QASPER_DIR,
    QUESTIONS_FILE,
    SPLIT,
    TOPIC_KEYWORDS,
    extract_questions,
    pdf_path,
    select_papers,
)

MANIFEST_FILE = "manifest.json"


def write_json(path, payload):
    """Write a JSON file, creating parent directories when needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def sha256_of(path):
    """Return the SHA-256 hex digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paper_record(paper):
    """Keep the QASPER fields that serve as clean reference text for each paper."""
    full_text = paper["full_text"]
    return {
        "id": paper["id"],
        "title": paper["title"],
        "abstract": paper["abstract"],
        "full_text": {
            "section_name": list(full_text["section_name"]),
            "paragraphs": [list(section) for section in full_text["paragraphs"]],
        },
    }


def download_pdf(session, url, destination):
    """Download a PDF to the destination and raise ValueError if it is not a PDF."""
    response = session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    if not response.content.startswith(PDF_SIGNATURE):
        raise ValueError("the response is not a PDF file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(response.content)


def fetch_pdf(session, paper):
    """Return a manifest entry for one paper, downloading its PDF if it is missing."""
    paper_id = paper["id"]
    path = PDF_DIR / f"{paper_id.replace('/', '_')}.pdf"
    url = ARXIV_PDF_URL.format(paper_id=paper_id)
    entry = {"id": paper_id, "title": paper["title"], "file": path.as_posix(), "source_url": url}

    if path.exists():
        entry["status"] = "cached"
    else:
        try:
            download_pdf(session, url, path)
            entry["status"] = "downloaded"
        except (requests.RequestException, ValueError) as error:
            entry.update(status="failed", error=str(error))
        time.sleep(REQUEST_DELAY_SECONDS)

    if path.exists():
        entry.update(bytes=path.stat().st_size, sha256=sha256_of(path))
    return entry


def fetch_all(papers):
    """Download every missing PDF politely and return the manifest entries."""
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    return [fetch_pdf(session, paper) for paper in papers]


def main():
    """Save the selected QASPER papers, questions and PDFs to the data directory."""
    dataset = load_dataset(DATASET_NAME, revision=DATASET_REVISION, split=SPLIT)
    papers = select_papers(dataset, NUM_PAPERS, MIN_QUESTIONS, TOPIC_KEYWORDS)
    questions = [
        question._asdict() for paper in papers for question in extract_questions(paper)
    ]

    write_json(QASPER_DIR / PAPERS_FILE, [paper_record(paper) for paper in papers])
    write_json(QASPER_DIR / QUESTIONS_FILE, questions)

    entries = fetch_all(papers)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": "QASPER paper ids are arXiv ids; PDFs come from arxiv.org",
        "papers": entries,
    }
    write_json(DATA_DIR / MANIFEST_FILE, manifest)

    failed = [entry["id"] for entry in entries if entry["status"] == "failed"]
    print(f"Papers: {len(papers)}  Questions: {len(questions)}")
    for status in ("downloaded", "cached", "failed"):
        print(f"{status.capitalize():<11}: {sum(1 for e in entries if e['status'] == status)}")
    if failed:
        print(f"Failed ids (download these manually into {PDF_DIR}/): {', '.join(failed)}")
    print(f"Saved to {DATA_DIR}/")


if __name__ == "__main__":
    main()