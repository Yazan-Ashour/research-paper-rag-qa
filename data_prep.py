import re
from collections import namedtuple

from datasets import load_dataset

DATASET_NAME = "allenai/qasper"
DATASET_REVISION = "refs/convert/parquet"
SPLIT = "validation"
NUM_PAPERS = 8
MIN_QUESTIONS = 4
MAX_CHUNK_WORDS = 250
TOPIC_KEYWORDS = (
    "retrieval",
    "question answering",
    "language model",
    "reading comprehension",
    "information extraction",
)

NOISE_PATTERN = re.compile(r"\b(?:BIBREF|FIGREF|TABREF|SECREF|INLINEFORM|DISPLAYFORM)\d*\b")
EMPTY_BRACKETS_PATTERN = re.compile(r"[\(\[]\s*[,;\s]*[\)\]]")
SPACE_BEFORE_PUNCT_PATTERN = re.compile(r"\s+([,.;:!?])")
WHITESPACE_PATTERN = re.compile(r"\s+")
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")
FLOAT_PREFIX = "FLOAT SELECTED"
REPEATED_COMMAS_PATTERN = re.compile(r"(?:\s*,){2,}")
COMMA_BEFORE_PERIOD_PATTERN = re.compile(r",\s*\.")


Chunk = namedtuple("Chunk", ["chunk_id", "paper_id", "title", "section", "text"])
Question = namedtuple(
    "Question",
    ["question_id", "paper_id", "text", "gold_answers", "gold_evidence", "unanswerable"],
)


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


def split_long_text(text, max_words):
    """Split text into sentence-aligned pieces of at most max_words words."""
    if word_count(text) <= max_words:
        return [text]

    pieces = []
    current = []
    current_words = 0
    for sentence in SENTENCE_PATTERN.split(text):
        sentence_words = word_count(sentence)
        if current and current_words + sentence_words > max_words:
            pieces.append(" ".join(current))
            current, current_words = [], 0
        current.append(sentence)
        current_words += sentence_words
    if current:
        pieces.append(" ".join(current))
    return pieces


def merge_pieces(pieces, max_words):
    """Greedily merge consecutive pieces into chunks of at most max_words words."""
    chunks = []
    current = []
    current_words = 0
    for piece in pieces:
        piece_words = word_count(piece)
        if current and current_words + piece_words > max_words:
            chunks.append(" ".join(current))
            current, current_words = [], 0
        current.append(piece)
        current_words += piece_words
    if current:
        chunks.append(" ".join(current))
    return chunks


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


def chunk_paper(paper, max_words=MAX_CHUNK_WORDS):
    """Convert a paper into section-aware chunks, starting with its abstract."""
    paper_id = paper["id"]
    title = paper["title"]
    sections = [("Abstract", [paper["abstract"]])]
    sections.extend(
        zip(paper["full_text"]["section_name"], paper["full_text"]["paragraphs"])
    )

    chunks = []
    for section_name, paragraphs in sections:
        pieces = []
        for paragraph in paragraphs:
            cleaned = clean_text(paragraph)
            if cleaned:
                pieces.extend(split_long_text(cleaned, max_words))
        for text in merge_pieces(pieces, max_words):
            chunks.append(
                Chunk(
                    chunk_id=f"{paper_id}-{len(chunks)}",
                    paper_id=paper_id,
                    title=title,
                    section=clean_text(section_name) or "Untitled",
                    text=text,
                )
            )
    return chunks


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


def build_corpus(max_words=MAX_CHUNK_WORDS):
    """Load QASPER and return the selected papers, their chunks, and test questions."""
    dataset = load_dataset(DATASET_NAME, revision=DATASET_REVISION, split=SPLIT)
    papers = select_papers(dataset, NUM_PAPERS, MIN_QUESTIONS, TOPIC_KEYWORDS)

    chunks = []
    questions = []
    for paper in papers:
        chunks.extend(chunk_paper(paper, max_words))
        questions.extend(extract_questions(paper))
    return papers, chunks, questions


papers, chunks, questions = build_corpus()

def main():
    """Build the corpus and print a short summary."""
    papers, chunks, questions = build_corpus()
    print(f"Papers: {len(papers)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Questions: {len(questions)}")
    print(f"Unanswerable questions: {sum(q.unanswerable for q in questions)}")
    print(f"Average chunk length: {sum(word_count(c.text) for c in chunks) / len(chunks):.0f} words")


if __name__ == "__main__":
    main()