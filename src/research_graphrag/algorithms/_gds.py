from neo4j import Session

DEFAULT_GRAPH_NAME = "citations"


def ensure_projection(session: Session, name: str = DEFAULT_GRAPH_NAME) -> None:
    """Drop any existing in-memory graph with this name and re-project fresh.

    Re-projecting each run picks up newly seeded papers without stale state.
    Uses UNDIRECTED orientation so both PageRank (directional by semantics) and
    Louvain (which works best undirected) can share a single projection.
    """
    session.run(
        "CALL gds.graph.drop($name, false) YIELD graphName",
        name=name,
    ).consume()
    session.run(
        """
        CALL gds.graph.project(
          $name,
          'Paper',
          {CITES: {orientation: 'UNDIRECTED'}}
        )
        """,
        name=name,
    ).consume()
