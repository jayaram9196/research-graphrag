import streamlit as st

from research_graphrag.retrieval.retrievers import (
    make_hybrid_retriever,
    make_text2cypher_retriever,
    make_vector_cypher_retriever,
    make_vector_retriever,
)

from _helpers import require_neo4j

st.set_page_config(page_title="Retrieve", page_icon="🔍", layout="wide")

st.title("🔍 Retrieve")
st.markdown(
    """
    Retrieve relevant papers using one of the four retrievers. **No LLM synthesis** —
    this page just shows you what the retriever found, so you can see the raw
    grounding data that the Ask / Reading List / Gaps pages use.

    | Mode | What it does |
    |------|--------------|
    | **vector** | Pure cosine similarity over abstract embeddings |
    | **graph** | Vector hit → expand through graph: concepts, authors, citations |
    | **hybrid** | Vector + fulltext (title + abstract keyword) combined |
    | **cypher** | Groq writes Cypher from your question, then runs it |
    """
)

if not require_neo4j():
    st.stop()

MODE_DESCRIPTIONS = {
    "graph": "VectorCypherRetriever — vector hit then graph expand (recommended for most questions).",
    "vector": "VectorRetriever — pure embedding similarity, fastest.",
    "hybrid": "HybridRetriever — blends vector score with fulltext score over title/abstract.",
    "cypher": "Text2CypherRetriever — best for analytical questions (e.g. 'highest cited', 'most prolific author').",
}

with st.form("retrieve_form"):
    question = st.text_input(
        "Question",
        value="text-to-image diffusion sampling techniques",
    )
    col1, col2 = st.columns([2, 1])
    with col1:
        mode = st.selectbox(
            "Mode",
            ["graph", "vector", "hybrid", "cypher"],
            format_func=lambda m: f"{m} — {MODE_DESCRIPTIONS[m].split(' — ')[0]}",
        )
    with col2:
        top_k = st.number_input("Top K", min_value=1, max_value=25, value=5)
    submit = st.form_submit_button("Run retrieval", type="primary")

st.caption(MODE_DESCRIPTIONS[mode])

if submit:
    factories = {
        "vector": make_vector_retriever,
        "graph": make_vector_cypher_retriever,
        "hybrid": make_hybrid_retriever,
        "cypher": make_text2cypher_retriever,
    }
    with st.spinner(f"Retrieving ({mode})..."):
        try:
            retriever = factories[mode]()
            if mode == "cypher":
                result = retriever.search(query_text=question)
            else:
                result = retriever.search(query_text=question, top_k=int(top_k))
        except Exception as exc:
            st.error(f"Retrieval failed: {exc}")
            st.stop()

    items = result.items[: int(top_k)]
    if not items:
        st.warning("No results.")
    else:
        if mode == "cypher" and len(result.items) > top_k:
            st.caption(
                f"Generated Cypher returned {len(result.items)} rows; showing first {top_k}."
            )
        for i, item in enumerate(items, 1):
            meta = item.metadata or {}
            oid = meta.get("openalex_id")
            with st.container(border=True):
                st.markdown(f"**#{i}**")
                st.text(item.content)
                if oid:
                    st.markdown(f"[🔗 https://openalex.org/{oid}](https://openalex.org/{oid})")
