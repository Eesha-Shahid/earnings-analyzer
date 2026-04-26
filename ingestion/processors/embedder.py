from sentence_transformers import SentenceTransformer
from config import Config
import numpy as np


class Embedder:
    def __init__(self):
        print(f"Loading embedding model: {Config.EMBEDDING_MODEL}")
        self.model = SentenceTransformer(Config.EMBEDDING_MODEL)
        print("Embedding model loaded.")

    def embed(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> list[list[float]]:
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    def embed_single(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def similarity(
        self,
        embedding_a: list[float],
        embedding_b: list[float],
    ) -> float:
        a = np.array(embedding_a)
        b = np.array(embedding_b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def score_relevance(
        self,
        query: str,
        chunks: list[str],
        threshold: float = 0.4,
    ) -> list[dict]:
        """
        Score chunks against a query.
        Used in the self-correction loop to filter weak retrievals.
        """
        query_embedding = self.embed_single(query)
        chunk_embeddings = self.embed(chunks)

        scored = []
        for i, (chunk, chunk_emb) in enumerate(zip(chunks, chunk_embeddings)):
            score = self.similarity(query_embedding, chunk_emb)
            if score >= threshold:
                scored.append({
                    "index": i,
                    "text": chunk,
                    "relevance_score": round(score, 4),
                })

        scored.sort(key=lambda x: x["relevance_score"], reverse=True)
        return scored


# Singleton
embedder = Embedder()
