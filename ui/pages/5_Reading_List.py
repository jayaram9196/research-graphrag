import streamlit as st

from research_graphrag.pipelines.reading_list import reading_list

from _helpers import require_neo4j

st.set_page_config(page_title="Reading List", page_icon="📚", layout="wide")

st.title("📚 Reading List")
st.markdown(
    """
    Produces a curated, chronologically-ordered reading list for a topic.

    1. Finds all `:Concept` nodes whose name matches the topic (substring, case-insensitive)
    2. Collects papers with `:ABOUT` edges to those concepts
    3. Scores each paper by `citation_count + 10 × local_in_degree`
    4. Sorts the top N by year
    5. **Groq writes a 2-3 sentence rationale per paper**
    """
)

if not require_neo4j():
    st.stop()

with st.form("reading_list_form"):
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        topic = st.text_input("Topic", value="diffusion")
    with col2:
        level = st.selectbox(
            "Level",
            ["beginner", "intermediate", "advanced"],
            index=1,
        )
    with col3:
        max_items = st.number_input("Max items", min_value=3, max_value=15, value=7)
    submit = st.form_submit_button("Build reading list", type="primary")

if submit:
    if not topic.strip():
        st.error("Topic is required.")
        st.stop()

    with st.spinner(f"Building reading list for {topic!r}..."):
        try:
            result = reading_list(topic, level=level, max_items=int(max_items))
        except Exception as exc:
            st.error(f"Reading list failed: {exc}")
            st.stop()

    st.subheader(f"Reading list — {topic!r} ({level})")
    st.markdown(result["answer"])

    if result["papers"]:
        st.subheader("Papers")
        for p in result["papers"]:
            st.markdown(
                f"- **[{p['id']}]** *{p['title']}* ({p['year']}) — "
                f"{p.get('citation_count', 0)} citations · "
                f"[OpenAlex](https://openalex.org/{p['id']})"
            )
