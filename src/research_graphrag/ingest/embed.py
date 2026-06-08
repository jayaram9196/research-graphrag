import os
from typing import Any, Protocol

from rich.console import Console

from ..clients.neo4j_client import get_session
from ..config import get_settings

console = Console()


class _BatchEncoder(Protocol):
    def encode(
        self, texts: list[str], batch_size: int = 32, show_progress_bar: bool = False
    ) -> Any: ...


class _FastEmbedAdapter:
    """Adapts fastembed.TextEmbedding to the SentenceTransformer.encode() interface."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def encode(
        self, texts: list[str], batch_size: int = 32, show_progress_bar: bool = False
    ) -> list[list[float]]:
        return [vec.tolist() for vec in self._model.embed(texts, batch_size=batch_size)]


def _load_encoder() -> _BatchEncoder:
    settings = get_settings()
    provider = settings.embedding_provider
    if provider == "sentence-transformers":
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(settings.embedding_model)
    if provider == "fastembed":
        # On Windows, HuggingFace's cache uses symlinks, which require Developer Mode
        # or admin rights. Without them the model download fails with WinError 1314
        # and leaves a corrupted cache. Disabling symlinks makes it copy files instead.
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        from fastembed import TextEmbedding

        return _FastEmbedAdapter(TextEmbedding(model_name=settings.embedding_model))
    raise ValueError(f"Unsupported embedding_provider: {provider}")


FETCH_PENDING_CYPHER = """
MATCH (p:Paper)
WHERE p.abstract IS NOT NULL AND p.embedding IS NULL
RETURN p.id AS id, p.abstract AS abstract
LIMIT $batch
"""

WRITE_EMBEDDINGS_CYPHER = """
UNWIND $rows AS row
MATCH (p:Paper {id: row.id})
CALL db.create.setNodeVectorProperty(p, 'embedding', row.embedding)
RETURN count(*) AS updated
"""


def embed_pending(batch_size: int | None = None, limit: int | None = None) -> dict[str, int]:
    settings = get_settings()
    size = batch_size or settings.batch_embedding_size

    console.print(f"[bold]Loading embedder:[/bold] {settings.embedding_model}")
    encoder = _load_encoder()

    total = 0
    while True:
        with get_session() as session:
            rows = session.execute_read(
                lambda tx: [dict(r) for r in tx.run(FETCH_PENDING_CYPHER, batch=size)]
            )
        if not rows:
            break

        texts = [r["abstract"] for r in rows]
        vectors = encoder.encode(texts, batch_size=size, show_progress_bar=False)
        payload = [
            {"id": r["id"], "embedding": [float(x) for x in vec]}
            for r, vec in zip(rows, vectors, strict=True)
        ]

        with get_session() as session:
            session.execute_write(
                lambda tx: tx.run(WRITE_EMBEDDINGS_CYPHER, rows=payload).consume()
            )

        total += len(payload)
        console.print(f"  embedded {len(payload)} (total: {total})")

        if limit is not None and total >= limit:
            break

    return {"embedded": total}
