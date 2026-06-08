import streamlit as st

from research_graphrag.pipelines.research_rag import ask as ask_impl

from _helpers import require_neo4j

st.set_page_config(page_title="Ask", page_icon="💬", layout="wide")

st.title("💬 Ask")
st.markdown(
    """
    Full GraphRAG Q&A. The pipeline:

    1. Embeds your question and hits the vector index
    2. For each top hit, traverses the graph to pull in concepts, authors, and 5 cited works
    3. Hands the enriched context to **Groq (llama-3.3-70b-versatile)** for synthesis
    4. Returns the answer with OpenAlex source links

    **Tips:** specific questions work better than generic ones
    (e.g. *"Trace the evolution from DDPM to Stable Diffusion"* beats *"Tell me about diffusion"*).
    Raise **top_k** if answers feel thin — more candidates give the LLM more to work with.
    """
)

if not require_neo4j():
    st.stop()

with st.form("ask_form"):
    question = st.text_input(
        "Question",
        value="Trace the evolution from DDPM to Stable Diffusion",
    )
    top_k = st.number_input("Top K (retrieval depth)", min_value=1, max_value=20, value=8)
    submit = st.form_submit_button("Ask", type="primary")

if submit:
    if not question.strip():
        st.error("Question is required.")
        st.stop()

    with st.spinner("Retrieving context and asking Groq..."):
        try:
            result = ask_impl(question, top_k=int(top_k))
        except Exception as exc:
            st.error(f"Ask failed: {exc}")
            st.stop()

    st.subheader("Answer")
    st.markdown(result["answer"])

    if result["sources"]:
        st.subheader("Sources")
        for s in result["sources"]:
            oid = s["openalex_id"]
            st.markdown(f"- [https://openalex.org/{oid}](https://openalex.org/{oid})")
