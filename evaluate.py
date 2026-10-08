from data_prep import MAX_CHUNK_WORDS, build_corpus
from reporting import save_report
from vector_store import VectorStore

TOP_K_VALUES = (1, 3, 5)
NGRAM_SIZE = 4
COVERAGE_THRESHOLD = 0.5
MODES = (("paper_scoped", True), ("global", False))
REPORT_FILE = "retrieval_evaluation.json"


def ngrams(text, size=NGRAM_SIZE):
    """Return the set of lowercase word n-grams in the text."""
    words = text.lower().split()
    return {tuple(words[i:i + size]) for i in range(len(words) - size + 1)}


def covers(chunk_text, evidence):
    """Return True if the chunk contains at least half of the evidence paragraph."""
    target = ngrams(evidence)
    if not target:
        return evidence.lower() in chunk_text.lower()
    return len(target & ngrams(chunk_text)) / len(target) >= COVERAGE_THRESHOLD


def first_hit_rank(store, question, scoped):
    """Return the rank of the first chunk covering any gold evidence, or None."""
    results = store.search(
        question.text,
        top_k=max(TOP_K_VALUES),
        paper_id=question.paper_id if scoped else None,
    )
    for rank, (chunk, _) in enumerate(results, start=1):
        if any(covers(chunk.text, evidence) for evidence in question.gold_evidence):
            return rank
    return None


def evaluate_questions(store, questions, scoped):
    """Return (question, first_hit_rank) pairs for questions that have gold evidence."""
    return [
        (question, first_hit_rank(store, question, scoped))
        for question in questions
        if question.gold_evidence
    ]


def count_hits(ranks):
    """Return the number of questions answered within each top-k value."""
    return {
        k: sum(1 for rank in ranks if rank is not None and rank <= k)
        for k in TOP_K_VALUES
    }


def hit_rates(store, questions, scoped):
    """Return the evaluated question count and hit counts per top-k value."""
    ranks = [rank for _, rank in evaluate_questions(store, questions, scoped)]
    return len(ranks), count_hits(ranks)


def summarize_mode(outcomes):
    """Build the report section for one evaluation mode."""
    ranks = [rank for _, rank in outcomes]
    hits = count_hits(ranks)
    total = len(outcomes)
    return {
        "questions_evaluated": total,
        "hit_counts": {f"hit@{k}": hits[k] for k in TOP_K_VALUES},
        "hit_rates": {f"hit@{k}": round(hits[k] / total, 4) for k in TOP_K_VALUES},
        "per_question": [
            {
                "question_id": question.question_id,
                "question": question.text,
                "first_hit_rank": rank,
            }
            for question, rank in outcomes
        ],
    }


def print_report(label, summary):
    """Print hit rates for one evaluation mode."""
    total = summary["questions_evaluated"]
    print(f"\n{label} ({total} questions with gold evidence)")
    for k in TOP_K_VALUES:
        key = f"hit@{k}"
        print(f"  Hit@{k}: {summary['hit_rates'][key]:.1%} ({summary['hit_counts'][key]}/{total})")


def main():
    """Evaluate retrieval in paper-scoped and global modes and save the report."""
    _, _, questions = build_corpus()
    store = VectorStore.load()

    modes = {}
    for label, scoped in MODES:
        outcomes = evaluate_questions(store, questions, scoped)
        modes[label] = summarize_mode(outcomes)
        print_report(label, modes[label])

    report = {
        "settings": {
            "embedding_model": store.model_name,
            "dense_weight": store.dense_weight,
            "max_chunk_words": MAX_CHUNK_WORDS,
            "num_chunks": len(store.chunks),
            "top_k_values": list(TOP_K_VALUES),
            "hit_definition": (
                f"A question is a hit if one retrieved chunk covers at least "
                f"{COVERAGE_THRESHOLD:.0%} of a gold evidence paragraph "
                f"({NGRAM_SIZE}-gram overlap)"
            ),
            "first_hit_rank": "Rank of the first covering chunk; null means a miss within the top 5",
        },
        "modes": modes,
    }
    path = save_report(report, REPORT_FILE)
    print(f"\nReport saved to {path}")


if __name__ == "__main__":
    main()