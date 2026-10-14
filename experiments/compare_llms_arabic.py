import os
import random
from statistics import mean

from dotenv import load_dotenv
from openai import OpenAI

from arabic_corpus import build_arabic_corpus
from arabic_text import arabic_ratio, normalize_arabic
from experiments.compare_llms import (
    CITATION_PATTERN,
    REQUEST_TIMEOUT_SECONDS,
    SAMPLE_SEED,
    best_scores,
    make_case,
    rate,
    summarize,
    timed_generate,
)
from rag_pipeline import MAX_TOKENS, TEMPERATURE, TOP_K, RAGPipeline, is_refusal
from reporting import save_report
from vector_store import INDEX_DIR, VectorStore, read_config

LLM_MODELS = ("llama3.2:3b", "gemma3:4b", "gemma4:e4b", "qwen2.5:3b")
SOURCE = "scan"
EXTRACTOR = "ocr_ara"
MAX_IN_SCOPE_QUESTIONS = 20
ARABIC_ANSWER_THRESHOLD = 0.5
REPORT_FILE = "llm_comparison_arabic.json"
OUT_OF_SCOPE_QUESTIONS = (
    "ما هي نقطة غليان الماء عند مستوى سطح البحر؟",
    "من كتب رواية دون كيخوته؟",
    "كيف أعدّ عجينة الخبز في المنزل؟",
    "ما هي عاصمة أستراليا؟",
    "ما الفرق بين لغة بايثون ولغة جافا؟",
)


def build_cases(store, questions):
    """Build the in-scope and out-of-scope Arabic cases with identical retrieved context for every model."""
    sample_size = min(MAX_IN_SCOPE_QUESTIONS, len(questions))
    sampled = random.Random(SAMPLE_SEED).sample(questions, sample_size)
    in_scope = [make_case(store, q.text, q.gold_answers, paper_id=q.paper_id) for q in sampled]
    out_of_scope = [make_case(store, text, []) for text in OUT_OF_SCOPE_QUESTIONS]
    return in_scope, out_of_scope


def score_answer(answer, gold_answers):
    """Return the best F1 and recall of an Arabic answer after normalizing diacritics and letter variants."""
    return best_scores(normalize_arabic(answer), [normalize_arabic(gold) for gold in gold_answers])


def evaluate_model(client, store, model, in_scope, out_of_scope):
    """Run one model over all Arabic cases and return its summary and per-question details."""
    pipeline = RAGPipeline(store, client=client, model=model)
    timed_generate(pipeline, in_scope[0])

    details = []
    for case in in_scope:
        answer, error, seconds = timed_generate(pipeline, case)
        f1, recall = score_answer(answer, case.gold_answers)
        details.append(
            {
                "question": case.question,
                "answer": answer,
                "gold_answers": case.gold_answers[:2],
                "f1": round(f1, 4),
                "recall": round(recall, 4),
                "refused": is_refusal(answer),
                "cited": bool(CITATION_PATTERN.search(answer)),
                "empty": not answer.strip(),
                "in_arabic": arabic_ratio(answer) >= ARABIC_ANSWER_THRESHOLD,
                "seconds": round(seconds, 2),
                "error": error,
            }
        )

    refusals = []
    for case in out_of_scope:
        answer, error, _ = timed_generate(pipeline, case)
        refusals.append({"question": case.question, "answer": answer, "refused": is_refusal(answer), "error": error})

    answered = [row for row in details if not row["refused"] and not row["empty"]]
    summary = summarize(details, refusals)
    summary["empty_answer_rate"] = rate(details, "empty")
    summary["arabic_answer_rate"] = rate(answered, "in_arabic") if answered else 0.0
    return {"summary": summary, "in_scope_details": details, "out_of_scope_details": refusals}


def print_summary(models, ranking):
    """Print a ranked comparison table."""
    print(
        f"\n{'Model':<14}{'F1':>7}{'Recall':>8}{'Arabic':>8}{'Cited':>7}{'FalseRef':>9}"
        f"{'TrueRef':>8}{'Empty':>7}{'Sec/ans':>8}{'Errors':>7}"
    )
    for name in ranking:
        s = models[name]["summary"]
        print(
            f"{name:<14}{s['mean_f1']:>7.3f}{s['mean_recall']:>8.3f}{s['arabic_answer_rate']:>8.0%}"
            f"{s['citation_rate']:>7.0%}{s['false_refusal_rate']:>9.0%}{s['correct_refusal_rate']:>8.0%}"
            f"{s['empty_answer_rate']:>7.0%}{s['avg_seconds_per_answer']:>8.1f}{s['errors']:>7}"
        )


def main():
    """Compare Ollama models on Arabic questions over OCR-extracted Arabic PDFs and save the report."""
    load_dotenv(override=True)
    client = OpenAI(
        base_url=os.environ["LLM_BASE_URL"],
        api_key=os.environ["LLM_API_KEY"],
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    config = read_config(INDEX_DIR)
    _, chunks, questions = build_arabic_corpus(SOURCE, EXTRACTOR)
    store = VectorStore(config["model_name"], dense_weight=1.0)
    store.build(chunks)
    in_scope, out_of_scope = build_cases(store, questions)

    installed = {model.id for model in client.models.list().data}
    skipped = [model for model in LLM_MODELS if model not in installed]
    available = [model for model in LLM_MODELS if model in installed]
    if not available:
        raise SystemExit("No candidate models are installed. Run 'ollama pull <name>'.")
    if skipped:
        print(f"Skipping models that are not installed: {', '.join(skipped)}")

    models = {}
    for model in available:
        print(f"Evaluating {model}...", flush=True)
        models[model] = evaluate_model(client, store, model, in_scope, out_of_scope)

    ranking = sorted(models, key=lambda name: models[name]["summary"]["mean_recall"], reverse=True)
    report = {
        "settings": {
            "embedding_model": config["model_name"],
            "corpus": f"Arabic article PDFs ({SOURCE}) read with {EXTRACTOR}",
            "retrieval": "dense only, paper-scoped, top 5 chunks, identical context for every model",
            "top_k": TOP_K,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "sample_seed": SAMPLE_SEED,
            "in_scope_questions": len(in_scope),
            "out_of_scope_questions": len(out_of_scope),
            "scoring": "F1 and recall use normalized tokens against the ARCD answer spans; read the saved answers too, because token overlap undervalues correct paraphrases",
        },
        "ranking_by_mean_recall": ranking,
        "skipped_models": skipped,
        "models": models,
    }
    path = save_report(report, REPORT_FILE)
    print_summary(models, ranking)
    print(f"\nReport saved to {path}")


if __name__ == "__main__":
    main()