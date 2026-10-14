from data_prep import (
    ARABIC_DIR,
    ARABIC_PDF_DIR,
    ARABIC_SCAN_DIR,
    ARABIC_TEXT_DIR,
    Chunk,
    Question,
    group_units,
    page_units,
    read_json,
)
from pdf_extraction import clean_page_text, extract_pages

ARTICLES_FILE = "articles.json"
QUESTIONS_FILE = "questions.json"
ARABIC_CHUNK_WORDS = 200
SOURCES = {"digital": ARABIC_PDF_DIR, "scan": ARABIC_SCAN_DIR}


def load_articles():
    """Load the list of selected Arabic articles."""
    return read_json(ARABIC_DIR / ARTICLES_FILE)


def load_questions():
    """Load the ARCD evaluation questions for the selected articles."""
    return [Question(**record) for record in read_json(ARABIC_DIR / QUESTIONS_FILE)]


def chunk_article(article, source, extractor, max_words=ARABIC_CHUNK_WORDS, overlap_words=0):
    """Extract one Arabic article PDF and split it into page-aware chunks."""
    path = SOURCES[source] / f"{article['id']}.pdf"
    pages = extract_pages(path, extractor, ARABIC_TEXT_DIR / source)
    units = []
    for number, raw_text in enumerate(pages, start=1):
        units.extend(page_units(number, clean_page_text(raw_text), max_words))
    return [
        Chunk(
            chunk_id=f"{article['id']}-{index}",
            paper_id=article["id"],
            title=article["title"],
            section=f"Page {group[0].page}",
            text=" ".join(unit.text for unit in group),
        )
        for index, group in enumerate(group_units(units, max_words, overlap_words))
    ]


def build_arabic_corpus(source, extractor, max_words=ARABIC_CHUNK_WORDS, overlap_words=0):
    """Return the articles, PDF chunks and questions for one source and extractor."""
    articles = load_articles()
    chunks = [
        chunk
        for article in articles
        for chunk in chunk_article(article, source, extractor, max_words, overlap_words)
    ]
    return articles, chunks, load_questions()