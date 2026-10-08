import os
import random
import re
import time
from collections import Counter, namedtuple
from statistics import mean

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

from data_prep import build_corpus
from rag_pipeline import MAX_TOKENS, NOT_FOUND_MESSAGE, TEMPERATURE, TOP_K, RAGPipeline
from reporting import save_report
from vector_store import VectorStore

LLM_MODELS = ("gemma3:1b", "llama3.2:3b", "qwen2.5:3b", "qwen3:8b")
MAX_IN_SCOPE_QUESTIONS = 20
SAMPLE_SEED = 42
REQUEST_TIMEOUT_SECONDS = 180
REPORT_FILE = "llm_comparison.json"
OUT_OF_SCOPE_QUESTIONS = (
    "Who won the 2018 FIFA World Cup?",
    "What is the capital of Australia?",
    "How do I bake sourdough bread?",
    "What is the boiling point of water at sea level?",
    "Who wrote the novel Don Quixote?",
)
METRIC_DEFINITIONS = {
    "mean_f1": "Token-overlap F1 against the best matching gold answer",
    "mean_recall": "Share of gold answer tokens found in the answer, best gold answer",
    "citation_rate": "Share of in-scope answers containing a [n] citation",
    "false_refusal_rate": "Share of in-scope answers where the model wrongly refused",
    "correct_refusal_rate": "Share of out-of-scope questions the model refused",
    "avg_seconds_per_answer": "Mean generation time per in-scope answer",
    "errors": "Number of failed API calls",
}

TOKEN_PATTERN = re.compile(r"\w+")
CITATION_PATTERN = re.compile(r"\[\d+\]")
REFUSAL_MARKER = NOT_FOUND_MESSAGE.lower().rstrip(".")

Case = namedtuple("Case", ["question", "context", "gold_answers"])


def token_scores(prediction, reference):
    """Return token-level F1 and recall of a prediction against one reference."""
    prediction_counts = Counter(TOKEN_PATTERN.findall(prediction.lower()))
    reference_counts = Counter(TOKEN_PATTERN.findall(reference.lower()))
    overlap = sum((prediction_counts & reference_counts).values())
    if overlap == 0:
        return 0.0, 0.0
    precision = overlap / sum(prediction_counts.values())
    recall = overlap / sum(reference_counts.values())
    return 2 * precision * recall / (precision + recall), recall


def best_scores(prediction, references):
    """Return the best F1 and best recall over all gold answers."""
    scores = [token_scores(prediction, reference) for reference in references]
    return max(f1 for f1, _ in scores), max(recall for _, recall in scores)


def is_refusal(answer):
    """Return True if the answer is the standard not-found refusal."""
    return REFUSAL_MARKER in answer.lower()


def make_case(store, text, gold_answers, paper_id=None):
    """Retrieve context for a question once so every model receives the same input."""
    results = store.search(text, top_k=TOP_K, paper_id=paper_id)
    return Case(text, RAGPipeline.build_context(results), gold_answers)


def build_cases(store, questions):
    """Build the in-scope and out-of-scope evaluation cases."""
    answerable = [question for question in questions if not question.unanswerable]
    sample_size = min(MAX_IN_SCOPE_QUESTIONS, len(answerable))
    sampled = random.Random(SAMPLE_SEED).sample(answerable, sample_size)
    in_scope = [
        make_case(store, q.text, q.gold_answers, paper_id=q.paper_id) for q in sampled
    ]
    out_of_scope = [make_case(store, text, []) for text in OUT_OF_SCOPE_QUESTIONS]
    return in_scope, out_of_scope


def timed_generate(pipeline, case):
    """Generate an answer and return the answer, any error message, and elapsed seconds."""
    start = time.perf_counter()
    try:
        answer, error = pipeline.generate(case.question, case.context), None
    except OpenAIError as exc:
        answer, error = "", str(exc)
    return answer, error, time.perf_counter() - start


def rate(rows, key):
    """Return the share of rows where the given key is truthy."""
    return round(mean(1.0 if row[key] else 0.0 for row in rows), 4)


def summarize(details, refusals):
    """Aggregate per-question rows into the model summary."""
    return {
        "mean_f1": round(mean(row["f1"] for row in details), 4),
        "mean_recall": round(mean(row["recall"] for row in details), 4),
        "citation_rate": rate(details, "cited"),
        "false_refusal_rate": rate(details, "refused"),
        "correct_refusal_rate": rate(refusals, "refused"),
        "avg_seconds_per_answer": round(mean(row["seconds"] for row in details), 2),
        "errors": sum(1 for row in details + refusals if row["error"]),
    }


def evaluate_model(store, client, model, in_scope, out_of_scope):
    """Run one model over all cases and return its summary and per-question details."""
    pipeline = RAGPipeline(store, client=client, model=model)
    timed_generate(pipeline, in_scope[0])

    details = []
    for case in in_scope:
        answer, error, seconds = timed_generate(pipeline, case)
        f1, recall = best_scores(answer, case.gold_answers)
        details.append(
            {
                "question": case.question,
                "answer": answer,
                "gold_answers": case.gold_answers[:2],
                "f1": round(f1, 4),
                "recall": round(recall, 4),
                "refused": is_refusal(answer),
                "cited": bool(CITATION_PATTERN.search(answer)),
                "seconds": round(seconds, 2),
                "error": error,
            }
        )

    refusals = []
    for case in out_of_scope:
        answer, error, _ = timed_generate(pipeline, case)
        refusals.append(
            {
                "question": case.question,
                "answer": answer,
                "refused": is_refusal(answer),
                "error": error,
            }
        )

    return {
        "summary": summarize(details, refusals),
        "in_scope_details": details,
        "out_of_scope_details": refusals,
    }


def print_summary(models, ranking):
    """Print a ranked comparison table."""
    print(
        f"\n{'Model':<18}{'F1':>8}{'Recall':>8}{'Cited':>8}"
        f"{'FalseRef':>10}{'TrueRef':>9}{'Sec/ans':>9}{'Errors':>8}"
    )
    for name in ranking:
        s = models[name]["summary"]
        print(
            f"{name:<18}{s['mean_f1']:>8.3f}{s['mean_recall']:>8.3f}"
            f"{s['citation_rate']:>8.0%}{s['false_refusal_rate']:>10.0%}"
            f"{s['correct_refusal_rate']:>9.0%}{s['avg_seconds_per_answer']:>9.1f}"
            f"{s['errors']:>8}"
        )


def main():
    """Compare Ollama models on identical retrieved contexts and save the report."""
    load_dotenv()
    client = OpenAI(
        base_url=os.environ["LLM_BASE_URL"],
        api_key=os.environ["LLM_API_KEY"],
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    _, _, questions = build_corpus()
    store = VectorStore.load()
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
        print(f"Evaluating {model}...")
        models[model] = evaluate_model(store, client, model, in_scope, out_of_scope)

    ranking = sorted(
        models, key=lambda name: models[name]["summary"]["mean_f1"], reverse=True
    )
    report = {
        "settings": {
            "embedding_model": store.model_name,
            "top_k": TOP_K,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "sample_seed": SAMPLE_SEED,
            "in_scope_questions": len(in_scope),
            "out_of_scope_questions": len(out_of_scope),
            "metric_definitions": METRIC_DEFINITIONS,
        },
        "ranking_by_mean_f1": ranking,
        "skipped_models": skipped,
        "models": models,
    }
    path = save_report(report, REPORT_FILE)
    print_summary(models, ranking)
    print(f"\nReport saved to {path}")


if __name__ == "__main__":
    main()