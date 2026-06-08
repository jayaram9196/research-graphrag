from typing import Any

from neo4j_graphrag.generation import GraphRAG

from ..retrieval.retrievers import get_llm, make_vector_cypher_retriever


def ask(question: str, top_k: int = 5) -> dict[str, Any]:
    """General Q&A: vector retrieval + graph expansion, then LLM synthesis."""
    retriever = make_vector_cypher_retriever()
    rag = GraphRAG(retriever=retriever, llm=get_llm())
    response = rag.search(
        query_text=question,
        retriever_config={"top_k": top_k},
        return_context=True,
    )
    sources: list[dict[str, Any]] = []
    if response.retriever_result is not None:
        for item in response.retriever_result.items:
            meta = item.metadata or {}
            if meta.get("openalex_id"):
                sources.append({"openalex_id": meta["openalex_id"]})
    return {"answer": response.answer, "sources": sources}
