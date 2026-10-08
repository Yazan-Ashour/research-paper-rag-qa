import itertools
import re

import numpy as np
from rank_bm25 import BM25Okapi

from data_prep import build_corpus, word_count
from evaluate import COVERAGE_THRESHOLD, ngrams
from reporting import save_report
from vector_store import VectorStore

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZES = (100, 150, 200, 250)
DENSE_WEIGHTS = (1.0, 0.7, 0.5, 0.3)
TOP_K_VALUES = (3, 5, 8)
MODES = (("paper_scoped", True), ("global", False))
BASELINE = {"max_chunk_words": 250, "dense_weight": 1.0, "top_k": 5}
TOP_RESULTS_SHOWN = 10
REPORT_FILE = "grid_search.json"
TOKEN_PATTERN = re.compile(r"\w+")


def tokenize(text):
    """Split text into lowercase word tokens."""
    return TOKEN_PATTERN.findall(text.lower())


def normalize_rows(matrix):
    """Min-max scale each row of a score matrix to the range 0 to 1."""
    low = matrix.min(axis=1, keepdims=True)
    spread = matrix.max(axis=1, keepdims=True) - low
    spread[spread == 0] = 1.0
    return (matrix - low) / spread


def score_matrices(store, chunks, questions):
    """Return normalized dense and BM25 score matrices of shape (questions, chunks)."""
    chunk_embeddings = store.encode([chunk.text for chunk in chunks])
    query_embeddings = store.encode([question.text for question in questions])
    dense = query_embeddings @ chunk_embeddings.T

    bm25 = BM25Okapi([tokenize(chunk.text) for chunk in chunks])
    lexical = np.array(
        [bm25.get_scores(tokenize(question.text)) for question in questions]
    )
    return normalize_rows(dense), normalize_rows(lexical)


def retrieve(scores, chunks, paper_id, top_k):
    """Return the top_k chunks by score, limited to one paper when paper_id is given."""
    if paper_id:
        in_paper = np.array([chunk.paper_id == paper_id for chunk in chunks])
        scores = np.where(in_paper, scores, -np.inf)
    order = np.argsort(-scores)[:top_k]
    return [chunks[position] for position in order if np.isfinite(scores[position])]


def evidence_found(retrieved, evidence):
    """Return True if the retrieved chunks together cover enough of the evidence."""
    target = ngrams(evidence)
    if not target:
        return any(evidence.lower() in chunk.text.lower() for chunk in retrieved)
    pool = set().union(*(ngrams(chunk.text) for chunk in retrieved))
    return len(target & pool) / len(target) >= COVERAGE_THRESHOLD


def evaluate_config(chunks, questions, dense, lexical, weight, top_k, scoped):
    """Return the hit rate and average context size for one configuration and mode."""
    hits = 0
    context_words = []
    for row, question in enumerate(questions):
        scores = weight * dense[row] + (1 - weight) * lexical[row]
        paper_id = question.paper_id if scoped else None
        retrieved = retrieve(scores, chunks, paper_id, top_k)
        context_words.append(sum(word_count(chunk.text) for chunk in retrieved))
        if any(evidence_found(retrieved, evidence) for evidence in question.gold_evidence):
            hits += 1
    return hits / len(questions), float(np.mean(context_words))


def run_grid(store):
    """Evaluate every combination of chunk size, dense weight, and top_k."""
    rows = []
    evaluated = 0
    for chunk_size in CHUNK_SIZES:
        _, chunks, questions = build_corpus(chunk_size)
        questions = [question for question in questions if question.gold_evidence]
        evaluated = len(questions)
        dense, lexical = score_matrices(store, chunks, questions)
        for weight, top_k in itertools.product(DENSE_WEIGHTS, TOP_K_VALUES):
            row = {
                "max_chunk_words": chunk_size,
                "dense_weight": weight,
                "top_k": top_k,
                "num_chunks": len(chunks),
            }
            for mode, scoped in MODES:
                hit_rate, context_words = evaluate_config(
                    chunks, questions, dense, lexical, weight, top_k, scoped
                )
                row[f"{mode}_hit_rate"] = round(hit_rate, 4)
                row[f"{mode}_context_words"] = round(context_words)
            row["score"] = round(
                (row["paper_scoped_hit_rate"] + row["global_hit_rate"]) / 2, 4
            )
            rows.append(row)
    return rows, evaluated


def print_summary(rows):
    """Print the best configurations in a ranked table."""
    print(
        f"\n{'Chunk':>6}{'DenseW':>8}{'TopK':>6}{'Scoped':>9}"
        f"{'Global':>9}{'Score':>8}{'CtxWords':>10}"
    )
    for row in rows[:TOP_RESULTS_SHOWN]:
        print(
            f"{row['max_chunk_words']:>6}{row['dense_weight']:>8}{row['top_k']:>6}"
            f"{row['paper_scoped_hit_rate']:>9.1%}{row['global_hit_rate']:>9.1%}"
            f"{row['score']:>8.1%}{row['paper_scoped_context_words']:>10}"
        )


def main():
    """Run the retrieval grid search and save the full report."""
    store = VectorStore(EMBEDDING_MODEL)
    rows, evaluated = run_grid(store)
    rows.sort(key=lambda row: (-row["score"], row["paper_scoped_context_words"]))
    baseline = next(
        row for row in rows if all(row[key] == value for key, value in BASELINE.items())
    )
    report = {
        "settings": {
            "embedding_model": EMBEDDING_MODEL,
            "evaluated_questions": evaluated,
            "chunk_sizes": list(CHUNK_SIZES),
            "dense_weights": list(DENSE_WEIGHTS),
            "top_k_values": list(TOP_K_VALUES),
            "score": "Mean of paper-scoped and global hit rate",
            "hit_definition": (
                "A question is a hit if the retrieved chunks together cover at "
                "least half of any gold evidence paragraph (4-gram overlap)"
            ),
            "hybrid_score": "dense_weight * dense + (1 - dense_weight) * bm25, min-max normalized",
            "tie_break": "Fewer context words wins",
        },
        "baseline": baseline,
        "best": rows[0],
        "results": rows,
    }
    path = save_report(report, REPORT_FILE)
    print_summary(rows)
    print(f"\nBaseline (250 words, dense only, k=5): score {baseline['score']:.1%}")
    print(f"Report saved to {path}")


if __name__ == "__main__":
    main()