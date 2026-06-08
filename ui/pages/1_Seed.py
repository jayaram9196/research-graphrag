import asyncio

import streamlit as st

from research_graphrag.config import get_settings
from research_graphrag.ingest.embed import embed_pending
from research_graphrag.ingest.seed import seed_topic

from _helpers import require_neo4j

st.set_page_config(page_title="Seed", page_icon="🌱", layout="wide")

st.title("🌱 Seed")
st.markdown(
    """
    Search OpenAlex for a topic and ingest the top papers into Neo4j, along with their
    authors, concepts, venues, and citation edges. Then embed every abstract.

    - **max-papers** caps how many unique papers to fetch (OpenAlex gives us titles, abstracts,
      authors, concepts, and referenced_works in one shot — no per-paper fetch needed).
    - **depth** controls BFS expansion over `referenced_works`. At `depth=1`, if the initial
      search already fills the cap, expansion is skipped.
    - All writes are `MERGE`-based, so re-running with the same topic is idempotent.
    """
)

if not require_neo4j():
    st.stop()

settings = get_settings()

with st.form("seed_form"):
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        topic = st.text_input(
            "Topic", value="diffusion models",
            help="Free-text query passed to OpenAlex /works?search=..."
        )
    with col2:
        max_papers = st.number_input(
            "Max papers", min_value=10, max_value=2000,
            value=settings.max_papers_per_seed, step=10,
        )
    with col3:
        depth = st.number_input(
            "Depth", min_value=0, max_value=3,
            value=settings.max_citation_depth,
        )
    skip_embed = st.checkbox("Skip embedding after ingest", value=False)
    submit = st.form_submit_button("Run seed", type="primary")

if submit:
    if not topic.strip():
        st.error("Topic is required.")
        st.stop()

    with st.spinner(f"Searching OpenAlex for {topic!r} and ingesting into Neo4j..."):
        try:
            stats = asyncio.run(seed_topic(topic, max_papers=int(max_papers), depth=int(depth)))
        except Exception as exc:
            st.error(f"Ingestion failed: {exc}")
            st.stop()

    st.success(f"Ingested {stats['total']} unique papers.")
    layer_counts = stats.get("layer_counts") or {}
    if layer_counts:
        cols = st.columns(len(layer_counts))
        for col, (layer, count) in zip(cols, sorted(layer_counts.items())):
            col.metric(f"Layer {layer}", count)

    if not skip_embed:
        with st.spinner("Embedding abstracts with fastembed..."):
            try:
                embed_stats = embed_pending()
            except Exception as exc:
                st.error(f"Embedding failed: {exc}")
                st.stop()
        st.success(f"Embedded {embed_stats['embedded']} new abstracts.")

    st.info(
        "Head to the **Stats** page to see the updated graph, or jump straight to "
        "**Ask** / **Retrieve** to query it."
    )
