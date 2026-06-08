from neo4j_graphrag.embeddings.base import Embedder

from ..ingest.embed import _load_encoder


class QueryEmbedder(Embedder):
    """Wraps the configured encoder (fastembed or sentence-transformers) to satisfy
    the neo4j-graphrag Embedder protocol (embed_query)."""

    def __init__(self) -> None:
        self._encoder = _load_encoder()

    def embed_query(self, text: str) -> list[float]:
        result = self._encoder.encode([text], batch_size=1, show_progress_bar=False)
        if hasattr(result, "tolist"):
            result = result.tolist()
        first = result[0]
        if hasattr(first, "tolist"):
            return first.tolist()
        return list(first)
