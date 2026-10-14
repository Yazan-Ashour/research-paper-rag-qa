import os

from dotenv import load_dotenv
from sentence_transformers import CrossEncoder

RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
N_CANDIDATES = 20
MAX_LENGTH = 1024  # Arabic uses more tokens per word; lower to 512 if it is too slow
BATCH_SIZE = 16


class Reranker:
    """Cross-encoder that rescores (question, chunk) pairs; scores are relevance probabilities in 0..1."""

    def __init__(self, model_name=RERANK_MODEL, max_length=MAX_LENGTH):
        """Load the cross-encoder once; reuse the instance for every question."""
        self.model_name = model_name
        self.model = CrossEncoder(model_name, max_length=max_length)
        self.cache = {}  # (question, chunk text) -> score, so repeated evaluations are cheap

    def score(self, query, chunks):
        """Return one relevance score per chunk, computing only the pairs not seen before."""
        missing = [chunk.text for chunk in chunks if (query, chunk.text) not in self.cache]
        if missing:
            scores = self.model.predict(
                [(query, text) for text in missing], batch_size=BATCH_SIZE
            )
            for text, value in zip(missing, scores):
                self.cache[(query, text)] = float(value)
        return [self.cache[(query, chunk.text)] for chunk in chunks]

    def rerank(self, query, results, top_k):
        """Reorder (chunk, score) results by cross-encoder score and keep the best top_k."""
        if not results:
            return []
        scores = self.score(query, [chunk for chunk, _ in results])
        ranked = sorted(zip(results, scores), key=lambda pair: pair[1], reverse=True)
        return [(chunk, score) for ((chunk, _), score) in ranked[:top_k]]


class RerankedStore:
    """Drop-in replacement for VectorStore: retrieve n_candidates, rerank, return the top_k."""

    def __init__(self, store, reranker, n_candidates=N_CANDIDATES):
        """Wrap a VectorStore; other attributes (chunks, model_name, ...) pass through to it."""
        self.store = store
        self.reranker = reranker
        self.n_candidates = n_candidates

    def search(self, query, top_k=5, paper_id=None):
        """Return the top_k chunks after reranking, with reranker scores."""
        candidates = self.store.search(
            query, top_k=max(top_k, self.n_candidates), paper_id=paper_id
        )
        return self.reranker.rerank(query, candidates, top_k)

    def __getattr__(self, name):
        """Delegate everything else (index, chunks, model_name, ...) to the wrapped store."""
        return getattr(self.store, name)


def wrap_store(store):
    """Wrap the store with a reranker if RERANK_MODEL is set in .env; otherwise return it unchanged."""
    load_dotenv()
    model_name = os.environ.get("RERANK_MODEL", "").strip()
    if not model_name:
        return store
    n_candidates = int(os.environ.get("RERANK_CANDIDATES", N_CANDIDATES))
    return RerankedStore(store, Reranker(model_name), n_candidates)