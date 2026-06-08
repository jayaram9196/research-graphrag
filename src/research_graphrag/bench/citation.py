import re
from typing import Any

from ..clients.neo4j_client import get_session
from ..pipelines.research_rag import ask

OPENALEX_ID_RE = re.compile(r"\bW\d{5,}\b")


def extract_ids(text: str) -> list[str]:
    return list(dict.fromkeys(OPENALEX_ID_RE.findall(text)))


def check_ids_in_graph(ids: list[str]) -> dict[str, dict[str, Any]]:
    if not ids:
        return {}
    with get_session() as session:
        rows = session.execute_read(
            lambda tx: list(
                tx.run(
                    """
                    UNWIND $ids AS id
                    OPTIONAL MATCH (p:Paper {id: id})
                    RETURN id,
                           p IS NOT NULL AS exists,
                           p.title IS NOT NULL AS has_title,
                           p.title AS title
                    """,
                    ids=ids,
                )
            )
        )
    return {r["id"]: dict(r) for r in rows}


def run_citation_suite(questions: list[str] | None = None, top_k: int = 6) -> dict[str, Any]:
    """For each question: run `ask`, extract OpenAlex IDs from the answer, check them
    against the graph. Counts ids_seen / in_graph / also_in_sources."""
    questions = questions or [
        "What are the foundational deep learning diffusion models for image generation?",
        "What is classifier-free guidance and why does it matter?",
        "Trace the evolution from DDPM to Stable Diffusion.",
    ]

    rows = []
    total_inline = 0
    total_inline_in_graph = 0
    total_inline_in_sources = 0
    total_sources = 0
    total_sources_in_graph = 0

    for q in questions:
        result = ask(q, top_k=top_k)
        answer_ids = extract_ids(result["answer"])
        source_ids = [s["openalex_id"] for s in result["sources"]]
        all_ids = list(dict.fromkeys(answer_ids + source_ids))
        checks = check_ids_in_graph(all_ids)

        inline_in_graph = [i for i in answer_ids if checks.get(i, {}).get("exists")]
        inline_in_sources = [i for i in answer_ids if i in source_ids]
        sources_in_graph = [i for i in source_ids if checks.get(i, {}).get("exists")]

        total_inline += len(answer_ids)
        total_inline_in_graph += len(inline_in_graph)
        total_inline_in_sources += len(inline_in_sources)
        total_sources += len(source_ids)
        total_sources_in_graph += len(sources_in_graph)

        rows.append({
            "question": q,
            "inline_cited_ids": answer_ids,
            "inline_in_graph": inline_in_graph,
            "inline_in_sources": inline_in_sources,
            "source_ids": source_ids,
            "sources_in_graph": sources_in_graph,
        })

    def _rate(num: int, denom: int) -> float | None:
        return (num / denom) if denom else None

    return {
        "questions": rows,
        "totals": {
            "ids_inline_in_answers": total_inline,
            "inline_verifiability_rate": _rate(total_inline_in_graph, total_inline),
            "inline_match_sources_rate": _rate(total_inline_in_sources, total_inline),
            "total_source_ids": total_sources,
            "sources_verifiability_rate": _rate(total_sources_in_graph, total_sources),
            "note": (
                "inline_* rates are None if the LLM doesn't write W-IDs in prose "
                "(default behavior — it cites by title/author). "
                "sources_verifiability_rate reflects the IDs our retriever supplies."
            ),
        },
    }
