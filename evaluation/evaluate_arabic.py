import re
import time
from statistics import mean

from arabic_corpus import ARABIC_CHUNK_WORDS, SOURCES, build_arabic_corpus, load_articles, load_questions
from arabic_text import normalize_arabic
from data_prep import ARABIC_TEXT_DIR
from pdf_extraction import clean_page_text, extract_pages
from reporting import save_report
from vector_store import INDEX_DIR, VectorStore, read_config

VARIANTS = (
    ("digital", "pymupdf"),
    ("digital", "pypdf"),
    ("digital", "ocr_ara"),
    ("scan", "pymupdf"),
    ("scan", "ocr_ara"),
)
MODES = (("paper_scoped", True), ("global", False))
TOP_K_VALUES = (1, 3, 5)
NGRAM_SIZE = 3
HYBRID_WEIGHT = 0.6
COVERAGE_THRESHOLD = 0.5
REPORT_FILE = "arabic_pipeline_evaluation.json"
TOKEN_PATTERN = re.compile(r"\w+")


def tokens(text):
    """Return normalized lowercase word tokens of an Arabic or mixed text."""
    return TOKEN_PATTERN.findall(normalize_arabic(text).lower())


def ngram_set(words):
    """Return the set of word n-grams in a token list."""
    return {tuple(words[i:i + NGRAM_SIZE]) for i in range(len(words) - NGRAM_SIZE + 1)}


def coverage(reference, pool):
    """Return the share of the reference n-grams found in the pool, or 0 if the reference is too short."""
    target = ngram_set(tokens(reference))
    return len(target & pool) / len(target) if target else 0.0


def reference_paragraphs(questions):
    """Return the distinct ARCD paragraphs of each article."""
    paragraphs = {}
    for question in questions:
        paragraphs.setdefault(question.paper_id, set()).update(question.gold_evidence)
    return {paper_id: sorted(texts) for paper_id, texts in paragraphs.items()}


def extract_variant(articles, source, extractor):
    """Extract every article without the cache and return the cleaned texts and the seconds spent."""
    texts, seconds = {}, 0.0
    for article in articles:
        start = time.perf_counter()
        pages = extract_pages(SOURCES[source] / f"{article['id']}.pdf", extractor, ARABIC_TEXT_DIR / source, refresh=True)
        seconds += time.perf_counter() - start
        texts[article["id"]] = " ".join(clean_page_text(page) for page in pages)
    return texts, seconds


def extraction_scores(texts, questions):
    """Score extracted text against the known ARCD paragraphs."""
    word_scores, gram_scores, by_paragraph = [], [], {}
    for paper_id, paragraphs in reference_paragraphs(questions).items():
        words = tokens(texts[paper_id])
        available_words, pool = set(words), ngram_set(words)
        for paragraph in paragraphs:
            reference_words = set(tokens(paragraph))
            if not reference_words:
                continue
            word_scores.append(len(reference_words & available_words) / len(reference_words))
            gram_scores.append(coverage(paragraph, pool))
            by_paragraph[(paper_id, paragraph)] = gram_scores[-1]
    reachable = [
        any(by_paragraph.get((q.paper_id, evidence), 0.0) >= COVERAGE_THRESHOLD for evidence in q.gold_evidence)
        for q in questions
    ]
    return {
        "word_recall": round(mean(word_scores), 4),
        "ngram_recall": round(mean(gram_scores), 4),
        "evidence_ceiling": round(mean(1.0 if flag else 0.0 for flag in reachable), 4),
    }


def covers(chunk_text, evidence):
    """Return True if the chunk covers at least half of the evidence paragraph."""
    return coverage(evidence, ngram_set(tokens(chunk_text))) >= COVERAGE_THRESHOLD


def hit_rates(store, questions, scoped):
    """Return hit@k rates for one search mode."""
    hits = dict.fromkeys(TOP_K_VALUES, 0)
    for question in questions:
        results = store.search(
            question.text,
            top_k=max(TOP_K_VALUES),
            paper_id=question.paper_id if scoped else None,
        )
        rank = next(
            (
                position
                for position, (chunk, _) in enumerate(results, start=1)
                if any(covers(chunk.text, evidence) for evidence in question.gold_evidence)
            ),
            None,
        )
        for k in TOP_K_VALUES:
            hits[k] += 1 if rank is not None and rank <= k else 0
    return {f"hit@{k}": round(hits[k] / len(questions), 4) for k in TOP_K_VALUES}


def retrieval_scores(store, chunks, questions, hybrid_weight):
    """Measure dense-only and hybrid retrieval over one set of chunks."""
    if not chunks:
        return {label: {mode: dict.fromkeys((f"hit@{k}" for k in TOP_K_VALUES), 0.0) for mode, _ in MODES} for label in ("dense", "hybrid")}
    store.build(chunks)
    scores = {}
    for label, weight in (("dense", 1.0), ("hybrid", hybrid_weight)):
        store.dense_weight = weight
        scores[label] = {mode: hit_rates(store, questions, scoped) for mode, scoped in MODES}
    return scores


def evaluate_variant(store, hybrid_weight, articles, questions, source, extractor):
    """Extract, chunk and evaluate one source and extractor combination."""
    texts, seconds = extract_variant(articles, source, extractor)
    _, chunks, _ = build_arabic_corpus(source, extractor)
    return {
        "source": source,
        "extractor": extractor,
        "extraction_seconds": round(seconds, 2),
        "chunks": len(chunks),
        "extraction": extraction_scores(texts, questions),
        "retrieval": retrieval_scores(store, chunks, questions, hybrid_weight),
    }


def print_summary(rows):
    """Print one line per source and extractor."""
    print(
        f"\n{'Source':<8}{'Extractor':<10}{'WordRec':>8}{'3gramRec':>9}{'Ceiling':>9}"
        f"{'Dense s@5':>10}{'Dense g@5':>10}{'Hyb s@5':>9}{'Hyb g@5':>9}{'Secs':>7}"
    )
    for row in rows:
        extraction, retrieval = row["extraction"], row["retrieval"]
        print(
            f"{row['source']:<8}{row['extractor']:<10}{extraction['word_recall']:>8.1%}{extraction['ngram_recall']:>9.1%}"
            f"{extraction['evidence_ceiling']:>9.1%}{retrieval['dense']['paper_scoped']['hit@5']:>10.1%}"
            f"{retrieval['dense']['global']['hit@5']:>10.1%}{retrieval['hybrid']['paper_scoped']['hit@5']:>9.1%}"
            f"{retrieval['hybrid']['global']['hit@5']:>9.1%}{row['extraction_seconds']:>7.1f}"
        )


def main():
    """Evaluate Arabic extraction and retrieval for digital PDFs and scans, with each extractor."""
    config = read_config(INDEX_DIR)
    store = VectorStore(config["model_name"], config["dense_weight"])
    articles, questions = load_articles(), load_questions()
    rows = []
    for source, extractor in VARIANTS:
        print(f"Evaluating {source} PDFs with {extractor}...", flush=True)
        rows.append(evaluate_variant(store, config["dense_weight"], articles, questions, source, extractor))

    report = {
        "settings": {
            "embedding_model": config["model_name"],
            "hybrid_dense_weight": config["dense_weight"],
            "articles": len(articles),
            "questions": len(questions),
            "chunk_words": ARABIC_CHUNK_WORDS,
            "ngram_size": NGRAM_SIZE,
            "ground_truth": "The ARCD paragraphs used to generate the PDFs, so extraction is scored against the exact source text",
            "metric_definitions": {
                "word_recall": "Mean share of each paragraph's distinct normalized words found in the extracted text",
                "ngram_recall": "Mean share of each paragraph's normalized word 3-grams found in the extracted text, which is sensitive to word order",
                "evidence_ceiling": "Share of questions whose paragraph has 3-gram recall of at least 0.5; the ceiling for retrieval",
                "hit@k": "A question is a hit if one retrieved chunk covers at least half of its paragraph",
            },
        },
        "results": rows,
    }
    path = save_report(report, REPORT_FILE)
    print_summary(rows)
    print(f"\nReport saved to {path}")


if __name__ == "__main__":
    main()