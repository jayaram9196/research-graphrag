import pandas as pd
import streamlit as st

from research_graphrag.algorithms.centrality import run_betweenness
from research_graphrag.algorithms.communities import detect_communities
from research_graphrag.algorithms.pagerank import run_pagerank

from _helpers import require_neo4j

st.set_page_config(page_title="Analyze", page_icon="📊", layout="wide")

st.title("📊 Analyze")
st.markdown(
    """
    Run Graph Data Science (GDS) algorithms on the citation network. **Requires the GDS
    plugin installed on your Neo4j instance.**

    | Algorithm | What it surfaces |
    |-----------|-----------------|
    | **PageRank** | Globally influential papers — like citation count, but weighted by the importance of the citer |
    | **Louvain** | Clusters papers into research communities based on citation structure |
    | **Betweenness** | Bridge papers sitting on many shortest paths — often surveys or cross-subfield works |
    """
)

if not require_neo4j():
    st.stop()

ALGOS = ["pagerank", "louvain", "betweenness"]

with st.form("analyze_form"):
    col1, col2, col3, col4 = st.columns([1.2, 2, 1, 1])
    with col1:
        algo = st.selectbox("Algorithm", ALGOS, index=0)
    with col2:
        concept = st.text_input(
            "Concept filter (pagerank/betweenness only)",
            value="",
            placeholder="e.g. diffusion, generative model",
        )
    with col3:
        limit = st.number_input("Limit", min_value=5, max_value=50, value=15)
    with col4:
        min_size = st.number_input("Min community size (Louvain)", min_value=2, max_value=50, value=5)
    submit = st.form_submit_button("Run algorithm", type="primary")

if submit:
    concept_val = concept.strip() or None
    with st.spinner(f"Running {algo}..."):
        try:
            if algo == "pagerank":
                rows = run_pagerank(concept=concept_val, limit=int(limit))
            elif algo == "betweenness":
                rows = run_betweenness(concept=concept_val, limit=int(limit))
            else:
                rows = detect_communities(limit=int(limit), min_size=int(min_size))
        except Exception as exc:
            st.error(f"Algorithm failed: {exc}")
            if "gds." in str(exc).lower():
                st.info(
                    "The GDS plugin isn't available on this Neo4j instance. In Neo4j "
                    "Desktop, open the instance → Plugins → install Graph Data Science, "
                    "then restart the instance."
                )
            st.stop()

    if not rows:
        st.warning("No results.")
        st.stop()

    if algo == "louvain":
        st.subheader(f"{len(rows)} communities (min size {int(min_size)})")
        for r in rows:
            with st.container(border=True):
                st.markdown(f"### Community {r['communityId']} — {r['size']} papers")
                for title, pid in zip(r["sample_titles"], r["sample_ids"]):
                    st.markdown(f"- [{pid}]({'https://openalex.org/' + pid}) — {title}")
    else:
        df = pd.DataFrame(rows)
        df["openalex"] = df["id"].apply(lambda i: f"https://openalex.org/{i}")
        st.dataframe(
            df[["title", "year", "citation_count", "score", "openalex"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "score": st.column_config.NumberColumn(format="%.4f"),
                "openalex": st.column_config.LinkColumn("OpenAlex"),
            },
        )
