import json
import os
import random
import re
import unicodedata
from statistics import mean

from dotenv import load_dotenv
from openai import OpenAI

from highlighting import CITATION_PATTERN
from arabic_text import arabic_ratio
from experiments.compare_llms import (
    REQUEST_TIMEOUT_SECONDS,
    SAMPLE_SEED,
    make_case,
    timed_generate,
)
from experiments.compare_llms_arabic import OUT_OF_SCOPE_QUESTIONS
from data_prep import QASPER_DIR, load_questions
from rag_pipeline import MAX_TOKENS, TEMPERATURE, TOP_K, RAGPipeline, is_refusal
from reporting import save_report
from vector_store import VectorStore

LLM_MODELS = ("llama3.2:3b", "gemma3:4b")
ARABIC_QUESTIONS_FILE = "questions_ar.json"
MAX_IN_SCOPE_QUESTIONS = 20
ARABIC_ANSWER_THRESHOLD = 0.3
SAMPLES_TO_PRINT = 3
REPORT_FILE = "llm_comparison_crosslingual.json"
ANCHOR_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[.\-][A-Za-z0-9]+)*")
ARABIC_ANSWER_THRESHOLD = 0.3
LATIN_RUN_PATTERN = re.compile(r"(?:[A-Za-z][A-Za-z'\-]*[ ,.]+){7,}")
DIGIT_TABLE = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹٫", "01234567890123456789."
)
METRIC_DEFINITIONS = {
    "arabic_answer_rate": "Share of non-refused answers written mostly in Arabic letters",
    "stray_script_rate": "Share of non-refused answers containing letters that are neither Arabic nor Latin (garbled output)",
    "citation_rate": "Share of non-refused answers containing a [n] citation",
    "anchor_recall": (
        "Mean share of the gold answer's numbers and acronyms (KBQA, F1, 0.5) found in the Arabic answer, "
        "over questions whose gold answer has any; refusals and empty answers count as 0"
    ),
    "false_refusal_rate": "Share of in-scope questions where the model wrongly refused",
    "correct_refusal_rate": "Share of out-of-scope questions the model refused",
    "empty_answer_rate": "Share of in-scope questions with an empty answer",
}


def share(rows, key):
    """Return the share of rows where the key is truthy, or 0 when there are no rows."""
    return round(mean(1.0 if row[key] else 0.0 for row in rows), 4) if rows else 0.0


def anchors(text):
    """Return the lowercase numbers and acronym-like terms (KBQA, F1, SemEval) in a text."""
    found = ANCHOR_PATTERN.findall(text.translate(DIGIT_TABLE))
    return {
        token.lower()
        for token in found
        if any(char.isdigit() for char in token) or sum(char.isupper() for char in token) >= 2
    }


def latin_tokens(text):
    """Return every lowercase Latin-script or numeric token in a text."""
    return {token.lower() for token in ANCHOR_PATTERN.findall(text.translate(DIGIT_TABLE))}


def anchor_recall(answer, gold_answers):
    """Return the best share of a gold answer's anchors found in the answer, or None if no gold answer has any."""
    available = latin_tokens(answer)
    scores = [len(gold & available) / len(gold) for gold in map(anchors, gold_answers) if gold]
    return max(scores) if scores else None


def has_stray_script(text):
    """Return True if the text has letters or marks that are neither Arabic nor Latin."""
    for char in text:
        if unicodedata.category(char)[0] in "LM":
            if not unicodedata.name(char, "").startswith(("ARABIC", "LATIN", "COMBINING", "GREEK")):
                return True
    return False


def load_translations():
    """Load the Arabic translations of the QASPER questions."""
    with open(QASPER_DIR / ARABIC_QUESTIONS_FILE, encoding="utf-8") as file:
        return json.load(file)


def build_cases(store, questions, translations):
    """Build Arabic questions over the English papers, plus Arabic out-of-scope questions."""
    answerable = [q for q in questions if q.question_id in translations and not q.unanswerable]
    sampled = random.Random(SAMPLE_SEED).sample(answerable, min(MAX_IN_SCOPE_QUESTIONS, len(answerable)))
    in_scope = [
        make_case(store, translations[q.question_id], q.gold_answers, paper_id=q.paper_id)
        for q in sampled
    ]
    out_of_scope = [make_case(store, text, []) for text in OUT_OF_SCOPE_QUESTIONS]
    return in_scope, out_of_scope, [q.text for q in sampled]


def evaluate_model(store, client, model, in_scope, english_texts, out_of_scope):
    """Run one model over all cases and return its summary and per-question details."""
    pipeline = RAGPipeline(store, client=client, model=model)
    timed_generate(pipeline, in_scope[0])

    details = []
    for case, english in zip(in_scope, english_texts):
        answer, error, seconds = timed_generate(pipeline, case)
        empty = not answer.strip()
        details.append(
            {
                "question": case.question,
                "question_en": english,
                "answer": answer,
                "gold_answers": case.gold_answers[:2],
                "refused": is_refusal(answer),
                "empty": empty,
                "cited": bool(CITATION_PATTERN.search(answer)),
                "in_arabic": arabic_ratio(answer) >= ARABIC_ANSWER_THRESHOLD and not LATIN_RUN_PATTERN.search(answer),
                "stray_script": has_stray_script(answer),
                "anchor_recall": anchor_recall(answer, case.gold_answers),
                "seconds": round(seconds, 2),
                "error": error,
            }
        )

    refusals = []
    for case in out_of_scope:
        answer, error, _ = timed_generate(pipeline, case)
        refusals.append(
            {"question": case.question, "answer": answer, "refused": is_refusal(answer), "error": error}
        )

    answered = [row for row in details if not row["refused"] and not row["empty"]]
    anchored = [row["anchor_recall"] or 0.0 for row in details if row["anchor_recall"] is not None]
    summary = {
        "arabic_answer_rate": share(answered, "in_arabic"),
        "stray_script_rate": share(answered, "stray_script"),
        "citation_rate": share(answered, "cited"),
        "anchor_recall": round(mean(anchored), 4) if anchored else 0.0,
        "anchor_questions": len(anchored),
        "false_refusal_rate": share(details, "refused"),
        "correct_refusal_rate": share(refusals, "refused"),
        "empty_answer_rate": share(details, "empty"),
        "avg_seconds_per_answer": round(mean(row["seconds"] for row in details), 2),
        "errors": sum(1 for row in details + refusals if row["error"]),
    }
    return {"summary": summary, "in_scope_details": details, "out_of_scope_details": refusals}


def print_summary(models):
    """Print one comparison line per model, then a few answers to read."""
    print(
        f"\n{'Model':<14}{'Arabic':>8}{'Stray':>7}{'Cited':>7}{'Anchor':>8}"
        f"{'FalseRef':>9}{'TrueRef':>8}{'Empty':>7}{'Sec/ans':>8}{'Errors':>7}"
    )
    for name, result in models.items():
        s = result["summary"]
        print(
            f"{name:<14}{s['arabic_answer_rate']:>8.0%}{s['stray_script_rate']:>7.0%}"
            f"{s['citation_rate']:>7.0%}{s['anchor_recall']:>8.2f}{s['false_refusal_rate']:>9.0%}"
            f"{s['correct_refusal_rate']:>8.0%}{s['empty_answer_rate']:>7.0%}"
            f"{s['avg_seconds_per_answer']:>8.1f}{s['errors']:>7}"
        )
    for name, result in models.items():
        print(f"\n--- {name}: first {SAMPLES_TO_PRINT} answers ---")
        for row in result["in_scope_details"][:SAMPLES_TO_PRINT]:
            print(f"Q: {row['question']}\n   ({row['question_en']})\nA: {row['answer']}\nGold: {row['gold_answers'][:1]}\n")


def main():
    """Compare Ollama models on Arabic questions over the English QASPER papers and save the report."""
    load_dotenv(override=True)
    client = OpenAI(
        base_url=os.environ["LLM_BASE_URL"],
        api_key=os.environ["LLM_API_KEY"],
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    store = VectorStore.load()
    in_scope, out_of_scope, english_texts = build_cases(store, load_questions(), load_translations())

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
        models[model] = evaluate_model(store, client, model, in_scope, english_texts, out_of_scope)

    report = {
        "settings": {
            "embedding_model": store.model_name,
            "dense_weight": store.dense_weight,
            "corpus": "English QASPER papers (saved index); the questions are the Arabic translations",
            "retrieval": f"as configured in the saved index, paper-scoped for in-scope questions, top {TOP_K} chunks, identical context for every model",
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "sample_seed": SAMPLE_SEED,
            "in_scope_questions": len(in_scope),
            "out_of_scope_questions": len(out_of_scope),
            "metric_definitions": METRIC_DEFINITIONS,
            "scoring_note": "Gold answers are English, so token F1 is not meaningful; read the saved answers to judge correctness",
        },
        "skipped_models": skipped,
        "models": models,
    }
    path = save_report(report, REPORT_FILE)
    print_summary(models)
    print(f"\nReport saved to {path}")


if __name__ == "__main__":
    main()