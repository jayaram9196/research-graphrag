from pathlib import Path
from typing import Any

from ..retrieval.retrievers import (
    make_hybrid_retriever,
    make_vector_cypher_retriever,
    make_vector_retriever,
)
from .gold_sets import load_gold_set

RETRIEVERS = {
    "graph": make_vector_cypher_retriever,
    "vector": make_vector_retriever,
    "hybrid": make_hybrid_retriever,
}


def _retrieve_ids(retriever, query: str, top_k: int) -> list[str]:
    result = retriever.search(query_text=query, top_k=top_k)
    ids: list[str] = []
    for item in result.items:
        meta = item.metadata or {}
        oid = meta.get("openalex_id")
        if oid:
            ids.append(oid)
    return ids


def evaluate_retriever(
    retriever_name: str,
    gold: dict[str, list[str]],
    top_k: int = 10,
) -> dict[str, Any]:
    """Compute recall@K and precision@K for one retriever over the gold set."""
    retriever = RETRIEVERS[retriever_name]()
    per_query = []
    recalls: list[float] = []
    precisions: list[float] = []

    for query, gold_ids in gold.items():
        retrieved = _retrieve_ids(retriever, query, top_k)
        gold_set = set(gold_ids)
        hits = [i for i in retrieved if i in gold_set]
        recall = len(hits) / len(gold_set) if gold_set else 0.0
        precision = len(hits) / len(retrieved) if retrieved else 0.0
        per_query.append({
            "query": query,
            "gold": gold_ids,
            "retrieved": retrieved,
            "hits": hits,
            "recall": recall,
            "precision": precision,
        })
        recalls.append(recall)
        precisions.append(precision)

    return {
        "retriever": retriever_name,
        "top_k": top_k,
        "mean_recall": sum(recalls) / len(recalls) if recalls else 0.0,
        "mean_precision": sum(precisions) / len(precisions) if precisions else 0.0,
        "per_query": per_query,
    }


def run_retrieval_suite(
    top_k: int = 10,
    gold_path: Path | None = None,
    retrievers: list[str] | None = None,
) -> dict[str, Any]:
    gold = load_gold_set(gold_path)
    if not gold:
        return {"error": "Gold set is empty. Seed benchmarks/gold_set.json first.", "results": []}
    names = retrievers or ["graph", "vector", "hybrid"]
    return {
        "gold_set_size": len(gold),
        "top_k": top_k,
        "results": [evaluate_retriever(name, gold, top_k=top_k) for name in names],
    }
