from typing import Any

from ..clients.neo4j_client import get_session
from ..retrieval.retrievers import get_llm

GAP_CYPHER = """
CALL () {
  MATCH (ca:Concept)
  WHERE toLower(ca.name) CONTAINS toLower($concept_a)
  RETURN collect(DISTINCT ca) AS concepts_a
}
CALL () {
  MATCH (cb:Concept)
  WHERE toLower(cb.name) CONTAINS toLower($concept_b)
  RETURN collect(DISTINCT cb) AS concepts_b
}
WITH concepts_a, concepts_b
UNWIND concepts_a AS ca
UNWIND concepts_b AS cb
OPTIONAL MATCH (p:Paper)-[:ABOUT]->(ca), (p)-[:ABOUT]->(cb)
WHERE p.title IS NOT NULL
WITH concepts_a, concepts_b,
     collect(DISTINCT {
       id: p.id,
       title: p.title,
       year: p.year,
       citation_count: p.citation_count,
       abstract: left(coalesce(p.abstract, ''), 500)
     }) AS papers
RETURN [c IN concepts_a | c.name] AS matched_a,
       [c IN concepts_b | c.name] AS matched_b,
       [x IN papers WHERE x.id IS NOT NULL] AS intersection_papers
"""

PROMPT_TEMPLATE = """You are a research gap analyst.

Two research areas:
- A: "{concept_a}" (matched concepts in graph: {matched_a})
- B: "{concept_b}" (matched concepts in graph: {matched_b})

Papers in this corpus tagged with BOTH concepts:

{papers_block}

Analyse the intersection:
1. How many papers exist at the intersection? Is this a well-explored area or a gap?
2. What themes do the existing papers cover? (only use the abstracts provided)
3. Which directions appear underexplored, based on what's NOT in the abstracts?

Keep the answer concrete and grounded in the papers shown. If the intersection is
empty, say so plainly and speculate carefully about why the gap exists.
"""


def find_gaps(concept_a: str, concept_b: str) -> dict[str, Any]:
    with get_session() as session:
        record = session.execute_read(
            lambda tx: tx.run(GAP_CYPHER, concept_a=concept_a, concept_b=concept_b).single()
        )

    matched_a = record["matched_a"] if record else []
    matched_b = record["matched_b"] if record else []
    papers = record["intersection_papers"] if record else []

    if not matched_a or not matched_b:
        missing = []
        if not matched_a:
            missing.append(f"'{concept_a}'")
        if not matched_b:
            missing.append(f"'{concept_b}'")
        return {
            "answer": (
                f"No concepts in the graph match {' or '.join(missing)}. "
                f"Try broader terms, or seed more papers covering these areas."
            ),
            "papers": [],
            "matched_a": matched_a,
            "matched_b": matched_b,
        }

    papers.sort(key=lambda p: p.get("year") or 0, reverse=True)

    if papers:
        lines = []
        for i, p in enumerate(papers, 1):
            lines.append(
                f"{i}. [{p['id']}] {p['title']} ({p.get('year')}) — "
                f"{p.get('citation_count', 0)} citations"
            )
            if p.get("abstract"):
                lines.append(f"   {p['abstract']}")
        papers_block = "\n".join(lines)
    else:
        papers_block = "(no papers in this corpus are tagged with both concepts)"

    prompt = PROMPT_TEMPLATE.format(
        concept_a=concept_a,
        concept_b=concept_b,
        matched_a=matched_a,
        matched_b=matched_b,
        papers_block=papers_block,
    )
    response = get_llm().invoke(prompt)
    return {
        "answer": response.content,
        "papers": papers,
        "matched_a": matched_a,
        "matched_b": matched_b,
    }
