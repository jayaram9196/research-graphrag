import streamlit as st

from research_graphrag.pipelines.gaps import find_gaps

from _helpers import require_neo4j

st.set_page_config(page_title="Gaps", page_icon="🔀", layout="wide")

st.title("🔀 Gaps")
st.markdown(
    """
    Analyze the intersection of two research concepts. Good for spotting **underexplored
    crossovers** — e.g. "diffusion" × "robotics", or "NLP" × "causal inference".

    - Finds `:Concept` nodes matching each term (case-insensitive substring)
    - Pulls papers with `:ABOUT` edges to **both** sets
    - Passes the intersection (with abstracts) to Groq for interpretation
    - Few papers → likely a gap; many → well-explored overlap
    """
)

if not require_neo4j():
    st.stop()

with st.form("gaps_form"):
    col1, col2 = st.columns(2)
    with col1:
        concept_a = st.text_input("Concept A", value="diffusion")
    with col2:
        concept_b = st.text_input("Concept B", value="generative")
    submit = st.form_submit_button("Find gaps", type="primary")

if submit:
    if not concept_a.strip() or not concept_b.strip():
        st.error("Both concepts are required.")
        st.stop()

    with st.spinner(f"Analyzing intersection of {concept_a!r} and {concept_b!r}..."):
        try:
            result = find_gaps(concept_a.strip(), concept_b.strip())
        except Exception as exc:
            st.error(f"Gaps failed: {exc}")
            st.stop()

    cols = st.columns(3)
    cols[0].metric("Matched concepts A", len(result["matched_a"]))
    cols[1].metric("Matched concepts B", len(result["matched_b"]))
    cols[2].metric("Intersection papers", len(result["papers"]))

    with st.expander("Matched concepts", expanded=False):
        st.markdown(f"**A** — {result['matched_a']}")
        st.markdown(f"**B** — {result['matched_b']}")

    st.subheader("Analysis")
    st.markdown(result["answer"])

    if result["papers"]:
        st.subheader("Intersection papers")
        for p in result["papers"]:
            st.markdown(
                f"- **[{p['id']}]** {p['title']} ({p.get('year')}) — "
                f"{p.get('citation_count', 0)} citations · "
                f"[OpenAlex](https://openalex.org/{p['id']})"
            )
