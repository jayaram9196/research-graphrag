from typing import Any

from neo4j import Record
from neo4j.graph import Node
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.llm.base import LLMInterface
from neo4j_graphrag.retrievers import (
    HybridRetriever,
    Text2CypherRetriever,
    VectorCypherRetriever,
    VectorRetriever,
)
from neo4j_graphrag.types import RetrieverResultItem

from ..clients.neo4j_client import get_driver
from ..config import get_settings
from ..schema.cypher import VECTOR_INDEX_NAME
from .embeddings import QueryEmbedder

FULLTEXT_INDEX_NAME = "paper_text"

PAPER_RETURN_PROPERTIES = ["id", "doi", "title", "abstract", "year", "citation_count"]

_BULKY_NODE_PROPS = {"embedding", "created_at"}


def _clean_node(node: Node) -> dict[str, Any]:
    return {k: v for k, v in dict(node).items() if k not in _BULKY_NODE_PROPS}


def _shorten_abstract(value: Any) -> Any:
    if isinstance(value, str) and len(value) > 400:
        return value[:400] + "…"
    return value


def paper_result_formatter(record: Record) -> RetrieverResultItem:
    """Format a retrieval record into a compact, readable string.

    - Drops node `embedding` and `created_at` fields.
    - Truncates long abstracts.
    - Pulls OpenAlex ID into metadata for downstream linking.

    Handles three shapes of records returned by different retrievers:
      * VectorCypherRetriever — scalar columns from `retrieval_query`
      * VectorRetriever with return_properties — `{"node": {...properties}, "score"}`
      * Retrievers without return_properties — `{"node": <Node>, "score"}`
    """
    flat: dict[str, Any] = {}
    openalex_id: str | None = None

    def _capture_id(candidate: Any) -> None:
        nonlocal openalex_id
        if openalex_id is None and isinstance(candidate, str):
            if candidate.startswith("W") and candidate[1:].isdigit():
                openalex_id = candidate

    for key in record.keys():
        val = record[key]
        if isinstance(val, Node):
            cleaned = {k: _shorten_abstract(v) for k, v in _clean_node(val).items()}
            flat[key] = cleaned
            _capture_id(cleaned.get("id"))
        elif isinstance(val, dict):
            cleaned = {
                k: _shorten_abstract(v)
                for k, v in val.items()
                if k not in _BULKY_NODE_PROPS
            }
            flat[key] = cleaned
            _capture_id(cleaned.get("id"))
        elif isinstance(val, list):
            flat[key] = [_clean_node(v) if isinstance(v, Node) else v for v in val]
        else:
            if key == "abstract":
                val = _shorten_abstract(val)
            flat[key] = val
            if key == "id":
                _capture_id(val)

    lines = [f"{k}={v}" for k, v in flat.items() if v not in (None, [], "")]
    content = "\n   ".join(lines)
    metadata: dict[str, Any] = {}
    if openalex_id:
        metadata["openalex_id"] = openalex_id
    return RetrieverResultItem(content=content, metadata=metadata)

GRAPH_RETRIEVAL_QUERY = """
WITH node AS paper, score
OPTIONAL MATCH (paper)-[:ABOUT]->(c:Concept)
OPTIONAL MATCH (paper)-[:AUTHORED_BY]->(a:Author)
OPTIONAL MATCH (paper)-[:CITES]->(cited:Paper)
WHERE cited.title IS NOT NULL
WITH paper, score,
     collect(DISTINCT c.name)[..10] AS concepts,
     collect(DISTINCT a.name)[..5] AS authors,
     collect(DISTINCT cited.title)[..5] AS cites
RETURN paper.id AS id,
       paper.doi AS doi,
       paper.title AS title,
       paper.year AS year,
       paper.citation_count AS citation_count,
       left(coalesce(paper.abstract, ''), 400) AS abstract_snippet,
       concepts,
       authors,
       cites,
       score
ORDER BY score DESC
"""

SCHEMA_DESCRIPTION = """
Node labels and properties:
- Paper {id: STRING, doi: STRING, title: STRING, abstract: STRING, year: INTEGER, citation_count: INTEGER, openalex_type: STRING}
- Author {id: STRING, name: STRING, orcid: STRING}
- Institution {id: STRING, name: STRING, country: STRING, type: STRING}
- Venue {id: STRING, name: STRING, type: STRING, issn: STRING}
- Concept {id: STRING, name: STRING, level: INTEGER, wikidata_url: STRING}

Relationships:
- (:Paper)-[:CITES]->(:Paper)
- (:Paper)-[:AUTHORED_BY {position: STRING}]->(:Author)
- (:Author)-[:AFFILIATED_WITH]->(:Institution)
- (:Paper)-[:ABOUT {score: FLOAT}]->(:Concept)
- (:Paper)-[:PUBLISHED_IN]->(:Venue)

Notes:
- Paper.id is the OpenAlex Work ID (e.g. 'W2741809807').
- Some Paper nodes are citation stubs that only have an id — filter with `WHERE paper.title IS NOT NULL` when you need full records.
"""


def get_llm() -> LLMInterface:
    settings = get_settings()
    provider = settings.llm_provider
    if provider == "openai":
        client_kwargs: dict[str, Any] = {"api_key": settings.openai_api_key}
        if settings.openai_base_url:
            client_kwargs["base_url"] = settings.openai_base_url
        return OpenAILLM(
            model_name=settings.llm_model,
            model_params={"temperature": 0.0},
            **client_kwargs,
        )
    raise ValueError(f"Unsupported llm_provider: {provider}")


def make_vector_retriever() -> VectorRetriever:
    return VectorRetriever(
        driver=get_driver(),
        index_name=VECTOR_INDEX_NAME,
        embedder=QueryEmbedder(),
        return_properties=PAPER_RETURN_PROPERTIES,
        result_formatter=paper_result_formatter,
    )


def make_vector_cypher_retriever() -> VectorCypherRetriever:
    return VectorCypherRetriever(
        driver=get_driver(),
        index_name=VECTOR_INDEX_NAME,
        embedder=QueryEmbedder(),
        retrieval_query=GRAPH_RETRIEVAL_QUERY,
        result_formatter=paper_result_formatter,
    )


def make_hybrid_retriever() -> HybridRetriever:
    return HybridRetriever(
        driver=get_driver(),
        vector_index_name=VECTOR_INDEX_NAME,
        fulltext_index_name=FULLTEXT_INDEX_NAME,
        embedder=QueryEmbedder(),
        return_properties=PAPER_RETURN_PROPERTIES,
        result_formatter=paper_result_formatter,
    )


def make_text2cypher_retriever() -> Text2CypherRetriever:
    return Text2CypherRetriever(
        driver=get_driver(),
        llm=get_llm(),
        neo4j_schema=SCHEMA_DESCRIPTION,
        result_formatter=paper_result_formatter,
    )
