from typing import Any

from ..clients.neo4j_client import get_session
from ..retrieval.retrievers import get_llm

SEMINAL_PAPERS_CYPHER = """
MATCH (c:Concept)
WHERE toLower(c.name) CONTAINS toLower($topic)
WITH collect(DISTINCT c) AS concepts
UNWIND concepts AS c
MATCH (p:Paper)-[:ABOUT]->(c)
WHERE p.title IS NOT NULL
OPTIONAL MATCH (p)<-[:CITES]-(citer:Paper)
WITH p, count(DISTINCT citer) AS in_deg
RETURN p.id AS id,
       p.title AS title,
       p.year AS year,
       p.citation_count AS citation_count,
       left(coalesce(p.abstract, ''), 600) AS abstract,
       in_deg AS local_cited_by
ORDER BY (coalesce(p.citation_count, 0) + in_deg * 10) DESC
LIMIT $limit
"""

FALLBACK_PAPERS_CYPHER = """
MATCH (p:Paper)
WHERE p.title IS NOT NULL
  AND (toLower(p.title) CONTAINS toLower($topic)
       OR toLower(coalesce(p.abstract, '')) CONTAINS toLower($topic))
OPTIONAL MATCH (p)<-[:CITES]-(citer:Paper)
WITH p, count(DISTINCT citer) AS in_deg
RETURN p.id AS id,
       p.title AS title,
       p.year AS year,
       p.citation_count AS citation_count,
       left(coalesce(p.abstract, ''), 600) AS abstract,
       in_deg AS local_cited_by
ORDER BY (coalesce(p.citation_count, 0) + in_deg * 10) DESC
LIMIT $limit
"""

PROMPT_TEMPLATE = """You are helping a researcher build a reading list on "{topic}".
Audience level: {level}.

Below are candidate papers ranked by a combination of global citation count and
local in-degree in the citation graph. They are listed in chronological order,
oldest first, so the researcher can read them in sequence.

{papers_block}

Write a reading list in numbered order (same order as above) with:
- The title, year, and OpenAlex ID on the first line
- A 2-3 sentence rationale explaining why the paper belongs in this sequence and
  what the reader should take away from it
- If the paper is clearly foundational vs. a survey vs. an application, say so

Be specific and tie each entry to the topic. Do not invent citations or facts
not supported by the abstract provided.
"""


def reading_list(topic: str, level: str = "intermediate", max_items: int = 7) -> dict[str, Any]:
    with get_session() as session:
        records = session.execute_read(
            lambda tx: [dict(r) for r in tx.run(SEMINAL_PAPERS_CYPHER, topic=topic, limit=max_items)]
        )
        if not records:
            records = session.execute_read(
                lambda tx: [
                    dict(r)
                    for r in tx.run(FALLBACK_PAPERS_CYPHER, topic=topic, limit=max_items)
                ]
            )

    if not records:
        return {"answer": f"No papers found matching '{topic}'.", "papers": []}

    records.sort(key=lambda r: r["year"] or 0)

    lines: list[str] = []
    for i, r in enumerate(records, 1):
        lines.append(
            f"{i}. [{r['id']}] {r['title']} ({r['year']}) — "
            f"cited {r['citation_count']} times globally, "
            f"{r['local_cited_by']} times within this corpus"
        )
        if r["abstract"]:
            lines.append(f"   Abstract: {r['abstract']}")

    prompt = PROMPT_TEMPLATE.format(
        topic=topic,
        level=level,
        papers_block="\n".join(lines),
    )

    llm = get_llm()
    response = llm.invoke(prompt)
    return {"answer": response.content, "papers": records}
