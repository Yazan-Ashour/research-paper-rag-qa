import re
import time
from statistics import mean

from data_prep import TEXT_DIR, clean_text, load_papers, load_questions, pdf_path
from pdf_extraction import EXTRACTORS, clean_page_text, extract_pages
from reporting import save_report

NGRAM_SIZE = 4
EVIDENCE_THRESHOLD = 0.5
RECOVERED_THRESHOLD = 0.9
REPORT_FILE = "extractor_comparison.json"
TOKEN_PATTERN = re.compile(r"\w+")


def tokenize(text):
    """Split text into lowercase word tokens."""
    return TOKEN_PATTERN.findall(text.lower())


def ngram_set(words):
    """Return the set of word n-grams in a token list."""
    return {tuple(words[i:i + NGRAM_SIZE]) for i in range(len(words) - NGRAM_SIZE + 1)}


def coverage(reference, extracted_ngrams):
    """Return the share of the reference n-grams found in the extracted text, or None if too short."""
    target = ngram_set(tokenize(reference))
    if not target:
        return None
    return len(target & extracted_ngrams) / len(target)


def reference_paragraphs(paper):
    """Return the cleaned QASPER paragraphs of a paper, starting with its abstract."""
    paragraphs = [paper["abstract"]]
    for section in paper["full_text"]["paragraphs"]:
        paragraphs.extend(section)
    return [text for text in map(clean_text, paragraphs) if text]


def evaluate_paper(paper, questions, extractor):
    """Extract one paper without the cache and score the text against the QASPER reference."""
    start = time.perf_counter()
    pages = extract_pages(pdf_path(paper["id"]), extractor, TEXT_DIR, refresh=True)
    seconds = time.perf_counter() - start

    text = " ".join(clean_page_text(page) for page in pages)
    extracted = ngram_set(tokenize(text))
    scores = [
        score
        for score in (coverage(paragraph, extracted) for paragraph in reference_paragraphs(paper))
        if score is not None
    ]
    own_questions = [q for q in questions if q.paper_id == paper["id"] and q.gold_evidence]
    recoverable = [
        any((coverage(evidence, extracted) or 0.0) >= EVIDENCE_THRESHOLD for evidence in q.gold_evidence)
        for q in own_questions
    ]
    return {
        "pages": len(pages),
        "characters": len(text),
        "seconds": seconds,
        "paragraph_scores": scores,
        "evidence_flags": recoverable,
    }


def summarize(results):
    """Aggregate per-paper results into the extractor summary."""
    scores = [score for result in results.values() for score in result["paragraph_scores"]]
    flags = [flag for result in results.values() for flag in result["evidence_flags"]]
    return {
        "paragraph_recall": round(mean(scores), 4),
        "paragraphs_fully_recovered": round(
            sum(score >= RECOVERED_THRESHOLD for score in scores) / len(scores), 4
        ),
        "questions_with_recoverable_evidence": round(sum(flags) / len(flags), 4),
        "pages": sum(result["pages"] for result in results.values()),
        "total_seconds": round(sum(result["seconds"] for result in results.values()), 2),
        "per_paper": {
            paper_id: {
                "paragraph_recall": round(mean(result["paragraph_scores"]), 4),
                "characters": result["characters"],
                "seconds": round(result["seconds"], 2),
            }
            for paper_id, result in results.items()
        },
    }


def evaluate_extractor(papers, questions, extractor):
    """Run one extractor over every paper and return its summary."""
    results = {paper["id"]: evaluate_paper(paper, questions, extractor) for paper in papers}
    return summarize(results)


def print_summary(extractors, ranking, skipped):
    """Print a ranked comparison table."""
    print(f"\n{'Extractor':<10}{'Recall':>9}{'Recovered':>11}{'Evidence':>10}{'Pages':>7}{'Seconds':>9}")
    for name in ranking:
        s = extractors[name]
        print(
            f"{name:<10}{s['paragraph_recall']:>9.1%}{s['paragraphs_fully_recovered']:>11.1%}"
            f"{s['questions_with_recoverable_evidence']:>10.1%}{s['pages']:>7}{s['total_seconds']:>9.1f}"
        )
    for name, reason in skipped.items():
        print(f"{name:<10}skipped: {reason}")


def main():
    """Compare PDF text extractors against the QASPER text and save the report."""
    papers = load_papers()
    questions = load_questions()
    extractors, skipped = {}, {}
    for name in EXTRACTORS:
        print(f"Extracting with {name}...", flush=True)
        try:
            extractors[name] = evaluate_extractor(papers, questions, name)
        except RuntimeError as error:
            skipped[name] = str(error)

    ranking = sorted(
        extractors,
        key=lambda name: (
            extractors[name]["questions_with_recoverable_evidence"],
            extractors[name]["paragraph_recall"],
        ),
        reverse=True,
    )
    report = {
        "settings": {
            "papers": len(papers),
            "evaluated_questions": sum(1 for q in questions if q.gold_evidence),
            "ngram_size": NGRAM_SIZE,
            "metric_definitions": {
                "paragraph_recall": "Mean share of each QASPER paragraph's word 4-grams found in the extracted text",
                "paragraphs_fully_recovered": f"Share of paragraphs with recall of at least {RECOVERED_THRESHOLD}",
                "questions_with_recoverable_evidence": (
                    f"Share of questions where some gold evidence paragraph has recall of at least "
                    f"{EVIDENCE_THRESHOLD}; this is the ceiling for retrieval hit rate"
                ),
            },
            "ground_truth": "QASPER's own clean paragraph text; citations were removed from the PDF text before scoring",
        },
        "ranking": ranking,
        "extractors": extractors,
        "skipped": skipped,
    }
    path = save_report(report, REPORT_FILE)
    print_summary(extractors, ranking, skipped)
    print(f"\nReport saved to {path}")


if __name__ == "__main__":
    main()