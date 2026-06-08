import streamlit as st

from research_graphrag.pipelines.lineage import trace_lineage

from _helpers import require_neo4j

st.set_page_config(page_title="Lineage", page_icon="🧬", layout="wide")

st.title("🧬 Lineage")
st.markdown(
    """
    Trace the citation path between two papers and narrate the intellectual evolution.

    - Uses Cypher's `shortestPath` on the `:CITES` graph (undirected)
    - Passes the sequence of papers (with abstracts) to Groq for narration
    - Stub papers (only `id`, no title) are acknowledged but not invented

    **You need OpenAlex IDs** (they look like `W4312933868`). Easiest source: the Ask page
    or the Retrieve page shows these as links.
    """
)

if not require_neo4j():
    st.stop()

with st.form("lineage_form"):
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        from_id = st.text_input("From (OpenAlex ID)", value="W4312933868")
    with col2:
        to_id = st.text_input("To (OpenAlex ID)", value="W3036167779")
    with col3:
        max_depth = st.number_input("Max depth", min_value=2, max_value=15, value=10)
    submit = st.form_submit_button("Trace lineage", type="primary")

if submit:
    if not from_id.strip() or not to_id.strip():
        st.error("Both IDs are required.")
        st.stop()

    with st.spinner("Finding shortest citation path..."):
        try:
            result = trace_lineage(from_id.strip(), to_id.strip(), max_depth=int(max_depth))
        except Exception as exc:
            st.error(f"Lineage failed: {exc}")
            st.stop()

    if result["papers"]:
        st.subheader(f"Path ({len(result['papers'])} papers)")
        for p in result["papers"]:
            st.markdown(
                f"- **[{p['id']}]** {p['title']} ({p.get('year')}) · "
                f"[OpenAlex](https://openalex.org/{p['id']})"
            )

    st.subheader("Narrative")
    st.markdown(result["answer"])
