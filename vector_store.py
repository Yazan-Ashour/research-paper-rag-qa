import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from data_prep import Chunk, build_corpus

import re
import numpy as np

from rank_bm25 import BM25Okapi

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_DIR = Path("index")
INDEX_FILE = "faiss.index"
CHUNKS_FILE = "chunks.json"
BATCH_SIZE = 64
DENSE_WEIGHT = 0.5
TOKEN_PATTERN = re.compile(r"\w+")

def tokenize(text):
    """Split text into lowercase word tokens."""
    return TOKEN_PATTERN.findall(text.lower())

def min_max(values):
    """Scale an array to the range 0 to 1, returning zeros for a constant array."""
    spread = values.max() - values.min()
    if spread == 0:
        return np.zeros_like(values, dtype="float32")
    return (values - values.min()) / spread


class VectorStore:
    """FAISS-backed store of chunk embeddings with cosine-similarity search."""

    def __init__(self, model_name=EMBEDDING_MODEL, dense_weight=DENSE_WEIGHT):
        """Load the embedding model and start with an empty index."""
        self.model_name = model_name
        self.dense_weight = dense_weight
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.bm25 = None
        self.chunks = []

    def encode(self, texts):
        """Return L2-normalized float32 embeddings for a list of texts."""
        embeddings = self.model.encode(
            texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > BATCH_SIZE,
        )
        return embeddings.astype("float32")
    def build_lexical_index(self):
        """Create the BM25 index over the stored chunks."""
        self.bm25 = BM25Okapi([tokenize(chunk.text) for chunk in self.chunks])


    def build(self, chunks):
        """Embed the chunks and index them for dense and BM25 search."""
        embeddings = self.encode([chunk.text for chunk in chunks])
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self.chunks = list(chunks)
        self.build_lexical_index()


    def search(self, query, top_k=5, paper_id=None):
        """Return the top_k chunks by hybrid dense and BM25 score, optionally limited to one paper."""
        total = self.index.ntotal
        scores, positions = self.index.search(self.encode([query]), total)
        dense = np.zeros(total, dtype="float32")
        dense[positions[0]] = scores[0]
        lexical = np.asarray(self.bm25.get_scores(tokenize(query)), dtype="float32")
        combined = self.dense_weight * min_max(dense) + (1 - self.dense_weight) * min_max(lexical)
        return [
            (self.chunks[position], float(combined[position]))
            for position in np.argsort(-combined)
            if paper_id is None or self.chunks[position].paper_id == paper_id
        ][:top_k]

    def save(self, directory=INDEX_DIR):
        """Persist the FAISS index and chunk metadata to a directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / INDEX_FILE))
        records = [chunk._asdict() for chunk in self.chunks]
        with open(directory / CHUNKS_FILE, "w", encoding="utf-8") as file:
            json.dump(records, file, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, directory=INDEX_DIR, model_name=EMBEDDING_MODEL):
        """Restore a store previously written by save."""
        directory = Path(directory)
        store = cls(model_name)
        store.index = faiss.read_index(str(directory / INDEX_FILE))
        with open(directory / CHUNKS_FILE, encoding="utf-8") as file:
            store.chunks = [Chunk(**record) for record in json.load(file)]
        store.build_lexical_index()
        return store


def main():
    """Build the index from the QASPER corpus, save it, and run a sample query."""
    _, chunks, questions = build_corpus()
    store = VectorStore()
    store.build(chunks)
    store.save()
    print(f"Indexed {store.index.ntotal} chunks, saved to '{INDEX_DIR}/'")

    sample = questions[0]
    print(f"\nQuery: {sample.text}")
    for chunk, score in store.search(sample.text, top_k=3):
        print(f"\n[{score:.3f}] {chunk.title} | {chunk.section}")
        print(chunk.text[:200])


if __name__ == "__main__":
    main()