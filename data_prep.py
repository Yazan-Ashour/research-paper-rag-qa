import json
import re
from collections import namedtuple
from pathlib import Path

from pdf_extraction import (
    SPACE_BEFORE_PUNCT_PATTERN,
    WHITESPACE_PATTERN,
    clean_page_text,
    extract_pages,
)

DATA_DIR = Path("data")
PDF_DIR = DATA_DIR / "pdfs"
QASPER_DIR = DATA_DIR / "qasper"
TEXT_DIR = DATA_DIR / "text"
ARABIC_DIR = DATA_DIR / "arabic"
ARABIC_PDF_DIR = ARABIC_DIR / "pdfs"
ARABIC_SCAN_DIR = ARABIC_DIR / "scans"
ARABIC_TEXT_DIR = ARABIC_DIR / "text"
PAPERS_FILE = "papers.json"
QUESTIONS_FILE = "questions.json"

DATASET_NAME = "allenai/qasper"
DATASET_REVISION = "refs/convert/parquet"
SPLIT = "validation"
NUM_PAPERS = 8
MIN_QUESTIONS = 4
MAX_CHUNK_WORDS = 250
OVERLAP_WORDS = 0
EXTRACTOR = "pymupdf"
MAX_BIBLIOGRAPHY_DENSITY = 0.05
TOPIC_KEYWORDS = (
    "retrieval",
    "question answering",
    "language model",
    "reading comprehension",
    "information extraction",
)

NOISE_PATTERN = re.compile(r"\b(?:BIBREF|FIGREF|TABREF|SECREF|INLINEFORM|DISPLAYFORM)\d*\b")
EMPTY_BRACKETS_PATTERN = re.compile(r"[\(\[]\s*[,;\s]*[\)\]]")
REPEATED_COMMAS_PATTERN = re.compile(r"(?:\s*,){2,}")
COMMA_BEFORE_PERIOD_PATTERN = re.compile(r",\s*\.")
SENTENCE_PATTERN = re.compile(r"(?<=[.!?\u061f])\s+")
BIBLIOGRAPHY_PATTERN = re.compile(
    r"\b(?:19|20)\d{2}[a-z]?\b|\bet al\b|\bproceedings\b|\barxiv\b|\bpreprint\b"
    r"|\bpages\b|\bpp\b|\bvol(?:ume)?\b|\bjournal\b|\bconference\b|\bworkshop\b"
    r"|\bassociation for computational\b|(?-i:\bIn [A-Z]{3,}\b)|\b\d{1,4}\s*[-\u2013]\s*\d{1,4}\b",
    re.IGNORECASE,
)
REFERENCES_HEADING_PATTERN = re.compile(
    r"^[ \t]*(?:\d+\.?[ \t]+)?(?:bibliographical[ \t]+references|references|bibliography)[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
FLOAT_PREFIX = "FLOAT SELECTED"

Chunk = namedtuple("Chunk", ["chunk_id", "paper_id", "title", "section", "text"])
Question = namedtuple(
    "Question",
    ["question_id", "paper_id", "text", "gold_answers", "gold_evidence", "unanswerable"],
)
Unit = namedtuple("Unit", ["page", "text", "words"])


def clean_text(text):
    """Remove dataset reference tokens and normalize whitespace and punctuation."""
    text = NOISE_PATTERN.sub("", text)
    text = EMPTY_BRACKETS_PATTERN.sub("", text)
    text = REPEATED_COMMAS_PATTERN.sub(",", text)
    text = COMMA_BEFORE_PERIOD_PATTERN.sub(".", text)
    text = SPACE_BEFORE_PUNCT_PATTERN.sub(r"\1", text)
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def word_count(text):
    """Return the number of whitespace-separated words in the text."""
    return len(text.split())


def select_papers(dataset, num_papers, min_questions, keywords):
    """Return the first papers matching the topic keywords with enough questions."""
    selected = []
    for paper in dataset:
        searchable = f"{paper['title']} {paper['abstract']}".lower()
        has_topic = any(keyword in searchable for keyword in keywords)
        if has_topic and len(paper["qas"]["question"]) >= min_questions:
            selected.append(paper)
        if len(selected) == num_papers:
            break
    return selected


def format_answer(annotation):
    """Convert a single QASPER answer annotation into answer text."""
    if annotation["unanswerable"]:
        return None
    if annotation["extractive_spans"]:
        return " ".join(clean_text(span) for span in annotation["extractive_spans"])
    if annotation["free_form_answer"]:
        return clean_text(annotation["free_form_answer"])
    return "Yes" if annotation["yes_no"] else "No"


def extract_questions(paper):
    """Build evaluation questions with gold answers and gold evidence for a paper."""
    qas = paper["qas"]
    questions = []
    for question_id, text, answers in zip(
        qas["question_id"], qas["question"], qas["answers"]
    ):
        annotations = answers["answer"]
        gold_answers = [
            answer
            for answer in map(format_answer, annotations)
            if answer is not None
        ]
        gold_evidence = sorted(
            {
                clean_text(evidence)
                for annotation in annotations
                for evidence in annotation["evidence"]
                if evidence and not evidence.startswith(FLOAT_PREFIX)
            }
        )
        questions.append(
            Question(
                question_id=question_id,
                paper_id=paper["id"],
                text=text,
                gold_answers=gold_answers,
                gold_evidence=gold_evidence,
                unanswerable=not gold_answers,
            )
        )
    return questions


def read_json(path):
    """Read a JSON file from the data directory with a helpful error if it is missing."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run 'python download_data.py' first.")
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def load_papers():
    """Load the saved QASPER paper records (id, title, abstract and reference text)."""
    return read_json(QASPER_DIR / PAPERS_FILE)


def load_questions():
    """Load the saved evaluation questions."""
    return [Question(**record) for record in read_json(QASPER_DIR / QUESTIONS_FILE)]


def pdf_path(paper_id):
    """Return the local PDF path for a paper id."""
    return PDF_DIR / f"{paper_id.replace('/', '_')}.pdf"


def page_units(page_number, text, max_words):
    """Split page text into sentence units, windowing any unit longer than max_words."""
    units = []
    for sentence in SENTENCE_PATTERN.split(text):
        words = sentence.split()
        for start in range(0, len(words), max_words):
            window = words[start:start + max_words]
            units.append(Unit(page_number, " ".join(window), len(window)))
    return units


def group_units(units, max_words, overlap_words):
    """Group consecutive units into chunks of at most max_words, repeating up to overlap_words."""
    groups = []
    start = 0
    while start < len(units):
        end = start
        words = 0
        while end < len(units) and (end == start or words + units[end].words <= max_words):
            words += units[end].words
            end += 1
        groups.append(units[start:end])
        if end >= len(units):
            break
        next_start = end
        carried = 0
        while next_start - 1 > start and carried + units[next_start - 1].words <= overlap_words:
            next_start -= 1
            carried += units[next_start].words
        start = next_start
    return groups


def is_bibliography(text):
    """Return True if the text is dense with years, venues and other reference-list markers."""
    matches = len(BIBLIOGRAPHY_PATTERN.findall(text))
    return matches / max(word_count(text), 1) > MAX_BIBLIOGRAPHY_DENSITY


def trim_references(pages):
    """Drop the reference list and return the kept (page_number, text) pairs and dropped word count."""
    kept = []
    dropped_words = 0
    in_references = False
    for number, text in enumerate(pages, start=1):
        if in_references:
            if is_bibliography(clean_page_text(text)):
                dropped_words += word_count(text)
                continue
            in_references = False
        match = None if in_references else REFERENCES_HEADING_PATTERN.search(text)
        if match is None:
            kept.append((number, text))
            continue
        kept.append((number, text[:match.start()]))
        dropped_words += word_count(text[match.start():])
        in_references = True
    return kept, dropped_words


def chunk_pdf(paper, extractor, max_words, overlap_words):
    """Extract a paper's PDF and return its page-aware chunks and the word count of dropped references."""
    path = pdf_path(paper["id"])
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run 'python download_data.py' first.")
    pages, dropped_words = trim_references(extract_pages(path, extractor, TEXT_DIR))
    units = []
    for number, raw_text in pages:
        units.extend(page_units(number, clean_page_text(raw_text), max_words))

    chunks = [
        Chunk(
            chunk_id=f"{paper['id']}-{index}",
            paper_id=paper["id"],
            title=paper["title"],
            section=f"Page {group[0].page}",
            text=" ".join(unit.text for unit in group),
        )
        for index, group in enumerate(group_units(units, max_words, overlap_words))
    ]
    return chunks, dropped_words


def build_corpus(max_words=MAX_CHUNK_WORDS, overlap_words=OVERLAP_WORDS, extractor=EXTRACTOR):
    """Load the saved papers and questions and return the papers, PDF chunks and questions."""
    papers = load_papers()
    chunks = []
    for paper in papers:
        paper_chunks, _ = chunk_pdf(paper, extractor, max_words, overlap_words)
        chunks.extend(paper_chunks)
    return papers, chunks, load_questions()


def main():
    """Build the corpus from the local PDFs and print a short summary."""
    papers = load_papers()
    questions = load_questions()
    chunks = []
    dropped_words = 0
    for paper in papers:
        paper_chunks, dropped = chunk_pdf(paper, EXTRACTOR, MAX_CHUNK_WORDS, OVERLAP_WORDS)
        chunks.extend(paper_chunks)
        dropped_words += dropped
        print(f"  {paper['id']}: {len(paper_chunks)} chunks, {dropped} reference words removed")
    print(f"Extractor: {EXTRACTOR}")
    print(f"Papers: {len(papers)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Reference words removed: {dropped_words}")
    print(f"Questions: {len(questions)}")
    print(f"Average chunk length: {sum(word_count(c.text) for c in chunks) / len(chunks):.0f} words")
    print(f"\nSample chunk [{chunks[1].title} | {chunks[1].section}]:\n{chunks[1].text[:300]}")


if __name__ == "__main__":
    main()