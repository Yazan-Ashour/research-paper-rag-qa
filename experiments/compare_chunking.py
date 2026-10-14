from collections import defaultdict
from statistics import mean

from data_prep import build_corpus, word_count
from evaluation.evaluate import COVERAGE_THRESHOLD, TOP_K_VALUES, covers, hit_rates, ngrams
from reporting import save_report
from vector_store import EMBEDDING_MODEL, VectorStore

CHUNK_SIZES = (150, 200, 250, 300)
OVERLAPS = (0, 25, 50, 75)
MODES = (("paper_scoped", True), ("global", False))
POOLED_K = 5
BASELINE = {"max_chunk_words": 250, "overlap_words": 0}
MAX_CONTEXT_WORDS = 1000
TOP_RESULTS_SHOWN = 10
REPORT_FILE = "chunking_comparison.json"


def chunk_ceiling(chunks, questions):
    """Return the share of questions whose evidence is covered by one chunk of its own paper."""
    by_paper = defaultdict(list)
    for chunk in chunks:
        by_paper[chunk.paper_id].append(chunk.text)
    return mean(
        1.0
        if any(
            covers(text, evidence)
            for text in by_paper[question.paper_id]
            for evidence in question.gold_evidence
        )
        else 0.0
        for question in questions
    )


def pool_covers(pool, evidence):
    """Return True if the pooled n-grams cover enough of the evidence paragraph."""
    target = ngrams(evidence)
    if not target:
        return False
    return len(target & pool) / len(target) >= COVERAGE_THRESHOLD


def pooled_hit_rate(store, questions, scoped):
    """Return the pooled hit rate of the top results and the average words sent to the model."""
    hits = 0
    context_words = []
    for question in questions:
        results = store.search(
            question.text,
            top_k=POOLED_K,
            paper_id=question.paper_id if scoped else None,
        )
        pool = set().union(*(ngrams(chunk.text) for chunk, _ in results))
        context_words.append(sum(word_count(chunk.text) for chunk, _ in results))
        if any(pool_covers(pool, evidence) for evidence in question.gold_evidence):
            hits += 1
    return hits / len(questions), mean(context_words)


def evaluate_config(store, max_words, overlap_words):
    """Rebuild the chunks with one configuration and measure retrieval quality."""
    _, chunks, questions = build_corpus(max_words, overlap_words)
    questions = [question for question in questions if question.gold_evidence]
    store.build(chunks)

    row = {
        "max_chunk_words": max_words,
        "overlap_words": overlap_words,
        "chunks": len(chunks),
        "avg_chunk_words": round(mean(word_count(chunk.text) for chunk in chunks), 1),
        "chunk_ceiling": round(chunk_ceiling(chunks, questions), 4),
    }
    for mode, scoped in MODES:
        total, hits = hit_rates(store, questions, scoped)
        pooled, context_words = pooled_hit_rate(store, questions, scoped)
        row[mode] = {
            "strict_hits": {f"hit@{k}": round(hits[k] / total, 4) for k in TOP_K_VALUES},
            "pooled_hit@5": round(pooled, 4),
            "context_words": round(context_words),
        }
    row["score"] = round(mean(row[mode]["pooled_hit@5"] for mode, _ in MODES), 4)
    return row


def matches(row, settings):
    """Return True if a result row has the given chunk size and overlap."""
    return all(row[key] == value for key, value in settings.items())


def print_summary(rows):
    """Print the best configurations in a ranked table."""
    print(
        f"\n{'Chunk':>6}{'Overlap':>8}{'Chunks':>7}{'Ceiling':>9}{'Scoped':>8}"
        f"{'Global':>8}{'Score':>8}{'Strict@5':>9}{'Context':>8}"
    )
    for row in rows[:TOP_RESULTS_SHOWN]:
        scoped, overall = row["paper_scoped"], row["global"]
        print(
            f"{row['max_chunk_words']:>6}{row['overlap_words']:>8}{row['chunks']:>7}"
            f"{row['chunk_ceiling']:>9.1%}{scoped['pooled_hit@5']:>8.1%}{overall['pooled_hit@5']:>8.1%}"
            f"{row['score']:>8.1%}{scoped['strict_hits']['hit@5']:>9.1%}{scoped['context_words']:>8}"
        )


def main():
    """Compare chunk sizes and overlaps and save the full report."""
    store = VectorStore(EMBEDDING_MODEL)
    rows = []
    for max_words in CHUNK_SIZES:
        for overlap_words in OVERLAPS:
            print(f"Chunk size {max_words}, overlap {overlap_words}...", flush=True)
            rows.append(evaluate_config(store, max_words, overlap_words))

    rows.sort(key=lambda row: (-row["score"], row["paper_scoped"]["context_words"]))
    baseline = next(row for row in rows if matches(row, BASELINE))
    within_budget = [
        row for row in rows if row["paper_scoped"]["context_words"] <= MAX_CONTEXT_WORDS
    ]
    report = {
        "settings": {
            "embedding_model": EMBEDDING_MODEL,
            "dense_weight": store.dense_weight,
            "chunk_sizes": list(CHUNK_SIZES),
            "overlaps": list(OVERLAPS),
            "max_context_words": MAX_CONTEXT_WORDS,
            "metric_definitions": {
                "chunk_ceiling": "Share of questions where a single chunk of the right paper covers the evidence; needs no retrieval",
                "strict_hits": "A question is a hit if one retrieved chunk covers the evidence (the evaluate.py metric)",
                "pooled_hit@5": "A question is a hit if the top 5 chunks together cover the evidence, as the model sees them",
                "context_words": "Average words sent to the model in paper-scoped mode",
                "score": "Mean of the paper-scoped and global pooled_hit@5",
            },
        },
        "baseline": baseline,
        "best": rows[0],
        "best_within_context_budget": within_budget[0] if within_budget else None,
        "results": rows,
    }
    path = save_report(report, REPORT_FILE)
    print_summary(rows)
    print(f"\nBaseline (250 words, no overlap): score {baseline['score']:.1%}")
    print(f"Report saved to {path}")


if __name__ == "__main__":
    main()