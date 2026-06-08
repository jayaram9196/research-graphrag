from typing import Any

from ..clients.neo4j_client import get_session
from ._gds import DEFAULT_GRAPH_NAME, ensure_projection

LOUVAIN_QUERY = """
CALL gds.louvain.stream($graph)
YIELD nodeId, communityId
WITH gds.util.asNode(nodeId) AS paper, communityId
WHERE paper.title IS NOT NULL
WITH communityId, collect(paper) AS papers
WHERE size(papers) >= $min_size
RETURN communityId,
       size(papers) AS size,
       [p IN papers | p.title][..5] AS sample_titles,
       [p IN papers | p.id][..5] AS sample_ids
ORDER BY size DESC
LIMIT $limit
"""


def detect_communities(limit: int = 10, min_size: int = 3) -> list[dict[str, Any]]:
    with get_session() as session:
        ensure_projection(session)
        return session.execute_read(
            lambda tx: [
                dict(r)
                for r in tx.run(
                    LOUVAIN_QUERY,
                    graph=DEFAULT_GRAPH_NAME,
                    limit=limit,
                    min_size=min_size,
                )
            ]
        )
