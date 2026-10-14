from statistics import mean

from experiments.compare_embeddings import arabic_questions
from data_prep import load_questions
from evaluation.evaluate import TOP_K_VALUES, hit_rates
from reporting import save_report
from vector_store import VectorStore

DENSE_WEIGHTS = (1.0, 0.8, 0.6, 0.5, 0.3, 0.0)
MODES = (("paper_scoped", True), ("global", False))
REPORT_FILE = "weight_sweep.json"


def evaluate_weight(store, weight, question_sets):
    """Measure retrieval for one dense weight on every question set."""
    store.dense_weight = weight
    rates = {}
    for language, questions in question_sets.items():
        rates[language] = {}
        for mode, scoped in MODES:
            total, hits = hit_rates(store, questions, scoped)
            rates[language][mode] = {f"hit@{k}": round(hits[k] / total, 4) for k in TOP_K_VALUES}
    by_language = {
        language: round(mean(rate for mode_rates in language_rates.values() for rate in mode_rates.values()), 4)
        for language, language_rates in rates.items()
    }
    return {
        "hit_rates": rates,
        "overall_by_language": by_language,
        "overall_hit_rate": round(mean(by_language.values()), 4),
    }


def print_summary(results, ranking):
    """Print the weights from best to worst."""
    print(
        f"\n{'Dense weight':>12}{'En scoped@5':>13}{'En global@5':>13}{'Ar scoped@5':>13}"
        f"{'Ar global@5':>13}{'Overall':>9}"
    )
    for weight in ranking:
        rates = results[str(weight)]["hit_rates"]
        print(
            f"{weight:>12}{rates['english']['paper_scoped']['hit@5']:>13.1%}"
            f"{rates['english']['global']['hit@5']:>13.1%}{rates['arabic']['paper_scoped']['hit@5']:>13.1%}"
            f"{rates['arabic']['global']['hit@5']:>13.1%}{results[str(weight)]['overall_hit_rate']:>9.1%}"
        )


def main():
    """Sweep the dense/BM25 weight on the saved index for English and Arabic questions."""
    store = VectorStore.load()
    questions = load_questions()
    question_sets = {
        "english": [question for question in questions if question.gold_evidence],
        "arabic": arabic_questions(questions),
    }
    results = {}
    for weight in DENSE_WEIGHTS:
        print(f"Dense weight {weight}...", flush=True)
        results[str(weight)] = evaluate_weight(store, weight, question_sets)

    ranking = sorted(
        DENSE_WEIGHTS, key=lambda weight: results[str(weight)]["overall_hit_rate"], reverse=True
    )
    report = {
        "settings": {
            "embedding_model": store.model_name,
            "chunks": len(store.chunks),
            "evaluated_questions": len(question_sets["english"]),
            "dense_weights": list(DENSE_WEIGHTS),
            "note": "dense_weight 1.0 is dense only and 0.0 is BM25 only; the saved index is reused, so no re-embedding happens",
            "overall_hit_rate": "Mean of the English and Arabic averages over hit@1, @3 and @5 in both modes",
        },
        "ranking": ranking,
        "weights": results,
    }
    path = save_report(report, REPORT_FILE)
    print_summary(results, ranking)
    print(f"\nReport saved to {path}")


if __name__ == "__main__":
    main()