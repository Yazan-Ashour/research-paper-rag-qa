import os
from collections import namedtuple

from dotenv import load_dotenv
from openai import OpenAI

from data_prep import build_corpus
from vector_store import VectorStore

TOP_K = 5
TEMPERATURE = 0.1
MAX_TOKENS = 400
NOT_FOUND_MESSAGE = "I could not find the answer in the provided documents."
SYSTEM_PROMPT = (
    "You are a precise assistant. Answer the question using only the numbered "
    "context passages. State the specific facts, names, and numbers the passages give "
    "in one to three sentences, and add passage numbers like [1] after the claims they support. "
    "Start directly with the answer and never repeat passage titles or headings. "
    "Passages may define an abbreviation in parentheses after its full name; "
    "use such definitions when asked what an abbreviation stands for. "
    f"If the passages do not contain the answer, reply exactly: {NOT_FOUND_MESSAGE}"
)

RAGAnswer = namedtuple("RAGAnswer", ["answer", "sources"])


class RAGPipeline:
    """Retrieval-augmented question answering over a VectorStore using NVIDIA NIM."""

    def __init__(self, store, client=None, model=None):
        """Create the pipeline, reading NIM settings from the environment by default."""
        load_dotenv()
        self.store = store
        self.client = client or OpenAI(
            base_url=os.environ["LLM_BASE_URL"],
            api_key=os.environ["LLM_API_KEY"],
        )
        self.model = model or os.environ["LLM_MODEL"]

    @staticmethod
    def build_context(results):
        """Format retrieved chunks as numbered context passages."""
        return "\n\n".join(
            f"[{number}] {chunk.text}"
            for number, (chunk, _) in enumerate(results, start=1)
        )

    def generate(self, question, context):
        """Ask the language model to answer the question from the context."""
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
        )
        return response.choices[0].message.content.strip()

    def ask(self, question, top_k=TOP_K, paper_id=None):
        """Retrieve relevant chunks and return the generated answer with its sources."""
        results = self.store.search(question, top_k=top_k, paper_id=paper_id)
        if not results:
            return RAGAnswer(NOT_FOUND_MESSAGE, [])
        answer = self.generate(question, self.build_context(results))
        if NOT_FOUND_MESSAGE.rstrip(".") in answer:
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