from typing import Any

from ..clients.neo4j_client import get_session
from ..retrieval.retrievers import get_llm

LINEAGE_CYPHER = """
MATCH (a:Paper {id: $from_id}), (b:Paper {id: $to_id})
MATCH path = shortestPath((a)-[:CITES*1..%d]-(b))
RETURN [n IN nodes(path) | {
  id: n.id,
  title: coalesce(n.title, '(no title — citation stub)'),
  year: n.year,
  abstract: left(coalesce(n.abstract, ''), 500),
  citation_count: n.citation_count
}] AS papers
"""

PROMPT_TEMPLATE = """You are narrating the intellectual lineage between two research papers.

Source paper: [{from_id}]
Target paper: [{to_id}]

The shortest path in the citation graph connecting them (oldest to newest where
possible) is below. Each step is a paper that links the two:

{papers_block}

Write a narrative in 3-5 paragraphs that traces how ideas evolved from the source
paper to the target paper, using these intermediate papers as stepping stones.
For each hop, explain what that paper contributed. If a paper is a "citation stub"
(no title/abstract), acknowledge the gap — do not invent content for it.
Finish by naming the core intellectual thread running through the chain.
"""


def trace_lineage(from_id: str, to_id: str, max_depth: int = 10) -> dict[str, Any]:
    query = LINEAGE_CYPHER % max_depth
    with get_session() as session:
        record = session.execute_read(
            lambda tx: tx.run(query, from_id=from_id, to_id=to_id).single()
        )

    if record is None:
        return {
            "answer": (
                f"No citation path found between {from_id} and {to_id} within "
                f"depth {max_depth}. The two papers may be in disconnected parts "
                f"of the graph, or you may need to seed more papers."
            ),
            "papers": [],
        }

    papers = record["papers"]
    papers.sort(key=lambda p: p.get("year") or 0)

    lines = []
    for i, p in enumerate(papers, 1):
        lines.append(f"{i}. [{p['id']}] {p['title']} ({p.get('year')})")
        if p.get("abstract"):
            lines.append(f"   Abstract: {p['abstract']}")

    prompt = PROMPT_TEMPLATE.format(
        from_id=from_id,
        to_id=to_id,
        papers_block="\n".join(lines),
    )
    response = get_llm().invoke(prompt)
    return {"answer": response.content, "papers": papers}
