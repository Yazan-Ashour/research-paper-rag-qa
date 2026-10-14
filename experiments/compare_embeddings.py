import json
import time
from statistics import mean

from data_prep import QASPER_DIR, build_corpus
from evaluation.evaluate import TOP_K_VALUES, hit_rates
from reporting import save_report
from vector_store import VectorStore

EMBEDDING_MODELS = (
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "intfloat/multilingual-e5-small",
    "BAAI/bge-m3",
)
ARABIC_QUESTIONS_FILE = "questions_ar.json"
MODES = (("paper_scoped", True), ("global", False))
REPORT_FILE = "embedding_comparison_multilingual.json"


def arabic_questions(questions):
    """Return the evaluation questions with their text replaced by the saved Arabic translation."""
    path = QASPER_DIR / ARABIC_QUESTIONS_FILE
    with open(path, encoding="utf-8") as file:
        translations = json.load(file)
    evaluated = [question for question in questions if question.gold_evidence]
    missing = [q.question_id for q in evaluated if q.question_id not in translations]
    if missing:
        raise ValueError(f"{len(missing)} questions have no Arabic translation in {path}")
    return [question._replace(text=translations[question.question_id]) for question in evaluated]


def evaluate_model(model_name, chunks, question_sets):
    """Index the chunks with one embedding model and measure dense retrieval for each language."""
    store = VectorStore(model_name, dense_weight=1.0)
    start = time.perf_counter()
    store.build(chunks)
    build_seconds = time.perf_counter() - start

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
        "dimension": store.index.d,
        "max_sequence_length": store.model.max_seq_length,
        "index_build_seconds": round(build_seconds, 2),
        "hit_rates": rates,
        "overall_by_language": by_language,
        "overall_hit_rate": round(mean(by_language.values()), 4),
    }


def print_summary(results, ranking):
    """Print a ranked comparison table."""
    print(
        f"\n{'Model':<58}{'English':>8}{'Arabic':>8}{'Overall':>9}"
        f"{'Ar scoped@5':>12}{'Ar global@5':>12}{'Build(s)':>9}"
    )
    for name in ranking:
        result = results[name]
        arabic = result["hit_rates"]["arabic"]
        print(
            f"{name:<58}{result['overall_by_language']['english']:>8.1%}"
            f"{result['overall_by_language']['arabic']:>8.1%}{result['overall_hit_rate']:>9.1%}"
            f"{arabic['paper_scoped']['hit@5']:>12.1%}{arabic['global']['hit@5']:>12.1%}"
            f"{result['index_build_seconds']:>9.1f}"
        )


def main():
    """Compare embedding models on English and Arabic questions and save the report."""
    _, chunks, questions = build_corpus()
    question_sets = {
        "english": [question for question in questions if question.gold_evidence],
        "arabic": arabic_questions(questions),
    }
    results = {}
    for name in EMBEDDING_MODELS:
        print(f"Evaluating {name}...", flush=True)
        results[name] = evaluate_model(name, chunks, question_sets)

    ranking = sorted(results, key=lambda name: results[name]["overall_hit_rate"], reverse=True)
    report = {
        "settings": {
            "chunks": len(chunks),
            "evaluated_questions": len(question_sets["english"]),
            "retrieval": "dense only (dense_weight = 1.0) so the models are compared without BM25",
            "top_k_values": list(TOP_K_VALUES),
            "arabic_questions": "Translations of the English questions; the evidence passages stay English (cross-lingual retrieval)",
            "overall_hit_rate": "Mean of the English and Arabic averages; each average covers hit@1, @3 and @5 in both modes",
        },
        "ranking": ranking,
        "models": results,
    }
    path = save_report(report, REPORT_FILE)
    print_summary(results, ranking)
    print(f"\nReport saved to {path}")


if __name__ == "__main__":
    main()