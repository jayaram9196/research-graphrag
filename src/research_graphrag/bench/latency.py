import statistics
import time
from contextlib import contextmanager
from typing import Any, Callable


@contextmanager
def timed(label: str, sink: list[dict[str, Any]]):
    """Context manager that records elapsed seconds into `sink`."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        sink.append({"label": label, "seconds": elapsed})


def summarise(samples: list[float]) -> dict[str, float]:
    """Compute p50, p95, mean, min, max over a list of elapsed seconds."""
    if not samples:
        return {"n": 0}
    s = sorted(samples)
    return {
        "n": len(samples),
        "mean": statistics.mean(s),
        "p50": s[len(s) // 2],
        "p95": s[min(len(s) - 1, int(len(s) * 0.95))],
        "min": s[0],
        "max": s[-1],
    }


def run_latency_suite(runs: int = 3) -> dict[str, Any]:
    """Time core commands across a few repetitions."""
    from ..algorithms.pagerank import run_pagerank
    from ..pipelines.research_rag import ask
    from ..retrieval.retrievers import make_vector_cypher_retriever

    results: dict[str, list[float]] = {"retrieve_graph": [], "ask": [], "pagerank": []}

    retriever = make_vector_cypher_retriever()
    for _ in range(runs):
        t0 = time.perf_counter()
        retriever.search(query_text="diffusion model sampling", top_k=5)
        results["retrieve_graph"].append(time.perf_counter() - t0)

    for _ in range(runs):
        t0 = time.perf_counter()
        ask("What are the foundational diffusion models?", top_k=5)
        results["ask"].append(time.perf_counter() - t0)

    for _ in range(runs):
        t0 = time.perf_counter()
        try:
            run_pagerank(limit=10)
            results["pagerank"].append(time.perf_counter() - t0)
        except Exception:
            break

    return {label: summarise(samples) for label, samples in results.items()}
