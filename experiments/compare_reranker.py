import json
import statistics
import time
from pathlib import Path

from data_prep import build_corpus
from evaluation.evaluate import MODES, covers, evaluate_questions, count_hits, TOP_K_VALUES
from reporting import save_report
from reranker import Reranker, RerankedStore
from vector_store import VectorStore

CANDIDATE_SIZES = (10, 20, 30)
RECALL_KS = (5, 10, 20, 30)
ARABIC_FILE = Path("data/qasper/questions_ar.json")
ID_KEYS = ("question_id", "id")
TEXT_KEYS = ("text_ar", "question_ar", "arabic", "text", "question")
REPORT_FILE = "reranker_comparison.json"
OUT_OF_SCOPE = [
    "Who won the 2018 FIFA World Cup?",
    "What is the capital of Australia?",
    "How do I bake sourdough bread?",
    "Who wrote the novel Don Quixote?",
    "What is the boiling point of water at sea level?",
]


def load_arabic_questions(questions):
    """Return the evaluation questions with their Arabic translations as text.

    The file format is a guess (a dict id -> text, or a list of dicts); adapt this
    to match how compare_embeddings.py loads questions_ar.json if it differs.
    Assumes Question is a namedtuple, like Chunk.
    """
    with open(ARABIC_FILE, encoding="utf-8") as file:
        raw = json.load(file)
    if isinstance(raw, dict):
        by_id = {str(key): value for key, value in raw.items()}
    else:
        by_id = {}
        for item in raw:
            question_id = next(item[key] for key in ID_KEYS if key in item)
            by_id[str(question_id)] = next(item[key] for key in TEXT_KEYS if key in item)
    return [
        question._replace(text=by_id[str(question.question_id)])
        for question in questions
        if str(question.question_id) in by_id
    ]


def candidate_recall(store, questions, scoped, k):
    """Return (hits, total): questions whose gold evidence is inside the first-stage top k."""
    hits = total = 0
    for question in questions:
        if not question.gold_evidence:
            continue
        total += 1
        results = store.search(
            question.text, top_k=k, paper_id=question.paper_id if scoped else None
        )
        if any(covers(chunk.text, e) for chunk, _ in results for e in question.gold_evidence):
            hits += 1
    return hits, total


def run(store, questions, scoped):
    """Evaluate one store and return (outcomes, seconds per question)."""
    start = time.perf_counter()
    outcomes = evaluate_questions(store, questions, scoped)
    return outcomes, (time.perf_counter() - start) / max(len(outcomes), 1)


def summarize(outcomes, seconds):
    """Hit counts, rates and latency for one configuration."""
    ranks = [rank for _, rank in outcomes]
    hits = count_hits(ranks)
    total = len(outcomes)
    return {
        "questions": total,
        "hit_counts": {f"hit@{k}": hits[k] for k in TOP_K_VALUES},
        "hit_rates": {f"hit@{k}": round(hits[k] / total, 4) for k in TOP_K_VALUES},
        "seconds_per_question": round(seconds, 2),
    }


def paired_changes(baseline, reranked):
    """Count questions the reranker fixed and broke at hit@5 (same question order in both)."""
    gained = sum(1 for (_, b), (_, r) in zip(baseline, reranked) if b is None and r is not None)
    lost = sum(1 for (_, b), (_, r) in zip(baseline, reranked) if b is not None and r is None)
    return {"gained": gained, "lost": lost}


def refusal_scores(store, questions):
    """Compare the top reranker score of in-scope and out-of-scope questions (global search)."""
    in_scope = [
        store.search(q.text, top_k=1)[0][1] for q in questions if q.gold_evidence
    ]
    out_scope = [store.search(text, top_k=1)[0][1] for text in OUT_OF_SCOPE]
    return {
        "in_scope_min": round(min(in_scope), 4),
        "in_scope_median": round(statistics.median(in_scope), 4),
        "out_of_scope_max": round(max(out_scope), 4),
        "out_of_scope_scores": [round(s, 4) for s in out_scope],
    }


def main():
    """Measure first-stage recall, then compare no reranker against several candidate sizes."""
    _, _, questions = build_corpus()
    question_sets = {"english": questions, "arabic": load_arabic_questions(questions)}
    store = VectorStore.load()
    reranker = Reranker()

    print("Step 1: first-stage recall (can a reranker help?)")
    recall = {}
    for language, language_questions in question_sets.items():
        for label, scoped in MODES:
            row = {}
            for k in RECALL_KS:
                hits, total = candidate_recall(store, language_questions, scoped, k)
                row[f"recall@{k}"] = round(hits / total, 4)
            recall[f"{language}_{label}"] = row
            print(f"  {language:8s} {label:13s}", row)

    print("\nStep 2: reranking")
    configs = {}
    for language, language_questions in question_sets.items():
        for label, scoped in MODES:
            baseline, seconds = run(store, language_questions, scoped)
            entry = {"no_reranker": summarize(baseline, seconds)}
            print(f"\n{language} / {label}")
            print(f"  no reranker : {entry['no_reranker']['hit_rates']}")
            for n in CANDIDATE_SIZES:
                reranked, seconds = run(
                    RerankedStore(store, reranker, n), language_questions, scoped
                )
                entry[f"rerank_{n}"] = summarize(reranked, seconds)
                entry[f"rerank_{n}"]["vs_baseline_hit@5"] = paired_changes(baseline, reranked)
                print(
                    f"  rerank {n:>2}   : {entry[f'rerank_{n}']['hit_rates']} "
                    f"{entry[f'rerank_{n}']['vs_baseline_hit@5']} "
                    f"{entry[f'rerank_{n}']['seconds_per_question']}s/q"
                )
            configs[f"{language}_{label}"] = entry

    print("\nStep 3: do reranker scores separate out-of-scope questions?")
    refusal = refusal_scores(RerankedStore(store, reranker, 20), questions)
    print(" ", refusal)

    report = {
        "settings": {
            "embedding_model": store.model_name,
            "reranker": reranker.model_name,
            "candidate_sizes": list(CANDIDATE_SIZES),
            "num_chunks": len(store.chunks),
            "note": "Same hit definition as evaluate.py. Scores after reranking are probabilities.",
        },
        "first_stage_recall": recall,
        "configurations": configs,
        "refusal_scores": refusal,
    }
    print(f"\nReport saved to {save_report(report, REPORT_FILE)}")


if __name__ == "__main__":
    main()