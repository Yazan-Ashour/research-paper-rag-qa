import os
import re
from collections import namedtuple

from dotenv import load_dotenv
from openai import OpenAI

from arabic_text import detect_language, normalize_arabic
from data_prep import build_corpus
from vector_store import VectorStore

TOP_K = 5
TEMPERATURE = 0.1
MAX_TOKENS = 400
NOT_FOUND_MESSAGES = {
    "en": "I could not find the answer in the provided documents.",
    "ar": "لم أجد الإجابة في المستندات المقدمة.",
}
NOT_FOUND_MESSAGE = NOT_FOUND_MESSAGES["en"]
LANGUAGE_NAMES = {"en": "English", "ar": "Arabic"}
ARABIC_REFUSAL_PATTERN = re.compile(r"(?:لم|لا)\s+(?:اجد|توجد|يوجد).{0,120}(?:المستندات|النصوص|المقاطع)")
ARABIC_STYLE_RULE = (
    "Answer in one or two sentences, start directly with the answer, "
    "and do not repeat the question or add an introduction. "
    "Paraphrase the passages in Arabic; never copy English sentences from them. "
    "If the passages do not mention the answer, say so even if you know it from general knowledge. "
)

RAGAnswer = namedtuple("RAGAnswer", ["answer", "sources"])


def build_system_prompt(language):
    """Return the system prompt for a question language, with a fixed sentence for refusals."""
    language_rule = ""
    if language != "en":
        language_rule = (
            f"Write the answer in {LANGUAGE_NAMES[language]}, whatever language the passages use. "
            f"{ARABIC_STYLE_RULE}"
        )
    return (
        "You are a precise assistant. Answer the question using only the numbered "
        "context passages. State the specific facts, names, and numbers the passages give "
        "in one to three sentences, and add passage numbers like [1] after the claims they support. "
        "Start directly with the answer and never repeat passage titles or headings. "
        "Passages may define an abbreviation in parentheses after its full name; "
        "use such definitions when asked what an abbreviation stands for. "
        f"{language_rule}"
        f"If the passages do not contain the answer, reply exactly: {NOT_FOUND_MESSAGES[language]}"
    )


def is_refusal(answer):
    """Return True if the answer says the documents do not contain the answer, in any supported language."""
    if any(m.rstrip(".").lower() in answer.lower() for m in NOT_FOUND_MESSAGES.values()):
        return True
    return bool(ARABIC_REFUSAL_PATTERN.search(normalize_arabic(answer)))


class RAGPipeline:
    """Retrieval-augmented question answering over a VectorStore using any OpenAI-compatible LLM endpoint."""

    def __init__(self, store, client=None, model=None):
        """Create the pipeline, reading the LLM settings from the environment by default."""
        load_dotenv(override=True)
        self.store = store
        self.client = client or OpenAI(
            base_url=os.environ["LLM_BASE_URL"],
            api_key=os.environ["LLM_API_KEY"],
        )
        self.model = model or os.environ["LLM_MODEL"]
        # Used for Arabic questions only; stays None when a model is passed explicitly, so comparison scripts test the model they name.
        self.model_ar = None if model else os.environ.get("LLM_MODEL_AR")

    @staticmethod
    def build_context(results):
        """Format retrieved chunks as numbered context passages."""
        return "\n\n".join(
            f"[{number}] {chunk.text}"
            for number, (chunk, _) in enumerate(results, start=1)
        )

    def generate(self, question, context):
        """Ask the language model to answer the question from the context, in the question's language."""
        language = detect_language(question)
        model = self.model_ar if language == "ar" and self.model_ar else self.model
        response = self.client.chat.completions.create(
            model=model,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "system", "content": build_system_prompt(language)},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
        )
        return (response.choices[0].message.content or "").strip()

    def ask(self, question, top_k=TOP_K, paper_id=None):
        """Retrieve relevant chunks and return the generated answer with its sources."""
        results = self.store.search(question, top_k=top_k, paper_id=paper_id)
        if not results:
            return RAGAnswer(NOT_FOUND_MESSAGES[detect_language(question)], [])
        answer = self.generate(question, self.build_context(results))
        if is_refusal(answer):
            return RAGAnswer(answer, [])
        return RAGAnswer(answer, results)


def print_result(question, result, gold_answers=None):
    """Print a question, the generated answer, the gold answers, and the sources."""
    print(f"\nQuestion: {question}")
    print(f"Answer: {result.answer}")
    if gold_answers:
        print(f"Gold: {gold_answers[:2]}")
    for number, (chunk, score) in enumerate(result.sources, start=1):
        print(f"  [{number}] {score:.3f} | {chunk.title} | {chunk.section}")


def main():
    """Run five paper-scoped QASPER questions and one out-of-scope question."""
    _, _, questions = build_corpus()
    pipeline = RAGPipeline(VectorStore.load())

    answerable = [question for question in questions if not question.unanswerable]
    for question in answerable[:5]:
        result = pipeline.ask(question.text, paper_id=question.paper_id)
        print_result(question.text, result, question.gold_answers)

    print_result(
        "Who won the 2018 FIFA World Cup?",
        pipeline.ask("Who won the 2018 FIFA World Cup?"),
    )


if __name__ == "__main__":
    main()