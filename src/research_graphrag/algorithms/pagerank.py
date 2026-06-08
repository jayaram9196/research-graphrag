from typing import Any

from ..clients.neo4j_client import get_session
from ._gds import DEFAULT_GRAPH_NAME, ensure_projection

_BASE_QUERY = """
CALL gds.pageRank.stream($graph)
YIELD nodeId, score
WITH gds.util.asNode(nodeId) AS paper, score
WHERE paper.title IS NOT NULL
"""

_CONCEPT_FILTER = """
AND EXISTS {
  MATCH (paper)-[:ABOUT]->(c:Concept)
  WHERE toLower(c.name) CONTAINS toLower($concept)
}
"""

_RETURN = """
RETURN paper.id AS id,
       paper.title AS title,
       paper.year AS year,
       paper.citation_count AS citation_count,
       score
ORDER BY score DESC
LIMIT $limit
"""


def run_pagerank(concept: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
    query = _BASE_QUERY + (_CONCEPT_FILTER if concept else "") + _RETURN
    params: dict[str, Any] = {"graph": DEFAULT_GRAPH_NAME, "limit": limit}
    if concept:
        params["concept"] = concept

    with get_session() as session:
        ensure_projection(session)
        return session.execute_read(
            lambda tx: [dict(r) for r in tx.run(query, **params)]
        )
