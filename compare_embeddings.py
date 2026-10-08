import time
from statistics import mean

from data_prep import build_corpus
from evaluate import TOP_K_VALUES, hit_rates
from reporting import save_report
from vector_store import VectorStore

EMBEDDING_MODELS = (
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    "BAAI/bge-small-en-v1.5",
)
MODES = (("paper_scoped", True), ("global", False))
KEY_METRIC = "hit@5"
REPORT_FILE = "embedding_comparison.json"


def evaluate_model(model_name, chunks, questions):
    """Index the chunks with one embedding model and measure retrieval hit rates."""
    store = VectorStore(model_name)
    start = time.perf_counter()
    store.build(chunks)
    build_seconds = time.perf_counter() - start

    hit_rate_by_mode = {}
    for mode, scoped in MODES:
        total, hits = hit_rates(store, questions, scoped)
        hit_rate_by_mode[mode] = {
            f"hit@{k}": round(hits[k] / total, 4) for k in TOP_K_VALUES
        }

    overall = mean(
        rate for rates in hit_rate_by_mode.values() for rate in rates.values()
    )
    return {
        "dimension": store.index.d,
        "max_sequence_length": store.model.max_seq_length,
        "index_build_seconds": round(build_seconds, 2),
        "hit_rates": hit_rate_by_mode,
        "overall_hit_rate": round(overall, 4),
    }


def print_summary(results, ranking):
    """Print a ranked comparison table."""
    print(
        f"\n{'Model':<50}{'Overall':>9}{'Scoped@5':>10}{'Global@5':>10}{'Build(s)':>10}"
    )
    for name in ranking:
        result = results[name]
        scoped = result["hit_rates"]["paper_scoped"][KEY_METRIC]
        global_ = result["hit_rates"]["global"][KEY_METRIC]
        print(
            f"{name:<50}{result['overall_hit_rate']:>9.1%}{scoped:>10.1%}"
            f"{global_:>10.1%}{result['index_build_seconds']:>10.1f}"
        )


def main():
    """Compare embedding models on retrieval quality and save the report."""
    _, chunks, questions = build_corpus()
    results = {}
    for name in EMBEDDING_MODELS:
        print(f"Evaluating {name}...")
        results[name] = evaluate_model(name, chunks, questions)

    ranking = sorted(
        results, key=lambda name: results[name]["overall_hit_rate"], reverse=True
    )
    report = {
        "settings": {
            "chunks": len(chunks),
            "evaluated_questions": sum(1 for q in questions if q.gold_evidence),
            "top_k_values": list(TOP_K_VALUES),
            "overall_hit_rate": "Mean of every hit@k value across both modes",
        },
        "ranking": ranking,
        "models": results,
    }
    path = save_report(report, REPORT_FILE)
    print_summary(results, ranking)
    print(f"\nReport saved to {path}")


if __name__ == "__main__":
    main()