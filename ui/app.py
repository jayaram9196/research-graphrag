"""Research Paper GraphRAG — Streamlit home page.

Run with: `streamlit run ui/app.py`
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Research GraphRAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ────────────────────────────────── HERO ──────────────────────────────────
st.markdown(
    """
    <div style="
        padding: 2.5rem 2rem;
        border-radius: 14px;
        background: linear-gradient(135deg, #1b2735 0%, #2b5876 50%, #4e4376 100%);
        color: white;
        margin-bottom: 1.5rem;
    ">
      <h1 style="margin:0 0 0.5rem 0; color:white; font-size: 2.4rem;">
        📚 Research Paper GraphRAG
      </h1>
      <p style="margin:0; font-size: 1.1rem; opacity: 0.9;">
        Build a citation graph of research papers, then ask questions
        grounded in real graph structure — not just vector similarity.
      </p>
      <p style="margin:0.6rem 0 0 0; font-size: 0.95rem; opacity: 0.75;">
        Neo4j + <code>neo4j-graphrag</code> · Groq LLM · fastembed · OpenAlex
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ───────────────────────── WHAT IT DOES (three cards) ─────────────────────
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("### 🌱 Ingest")
    st.write(
        "Search OpenAlex for any topic and BFS-ingest the papers plus their "
        "citation graph. Every write is idempotent via `MERGE`."
    )
with c2:
    st.markdown("### 🔍 Retrieve")
    st.write(
        "Four retrievers: pure vector, vector+graph expansion, hybrid (vector + "
        "fulltext), or text-to-cypher where Groq writes Cypher from your question."
    )
with c3:
    st.markdown("### 💬 Synthesize")
    st.write(
        "Four LLM-backed pipelines: general Q&A, chronological reading lists, "
        "citation lineage narration, and gap analysis between two concepts."
    )

st.divider()

# ─────────────────────── ARCHITECTURE DIAGRAM ───────────────────────────
st.subheader("How it works")

st.graphviz_chart(
    """
    digraph G {
        rankdir=LR;
        bgcolor="transparent";
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#2b5876", fillcolor="#e3edf7", fontcolor="#1b2735"];
        edge [color="#4e4376", arrowsize=0.7];

        OpenAlex [fillcolor="#fce8b2", label="OpenAlex API\\n(240M works)"];
        Seed [label="seed_topic()\\nBFS ingest"];
        Neo4j [fillcolor="#cfe8cf", label="Neo4j\\nPaper • Author • Concept\\nCITES • ABOUT • …"];
        Embed [label="embed_pending()\\nfastembed → 384-dim"];
        VecIdx [fillcolor="#e8d4f5", label="Vector Index\\npaper_abstracts"];
        Retrievers [label="VectorCypher\\nRetriever"];
        Groq [fillcolor="#f4c7b8", label="Groq LLM\\n(llama-3.3-70b)"];
        Answer [fillcolor="#d6eadf", label="Grounded answer\\n+ OpenAlex links"];
        GDS [fillcolor="#d4e6f1", label="GDS Algorithms\\nPageRank • Louvain •\\nBetweenness"];

        OpenAlex -> Seed -> Neo4j;
        Neo4j -> Embed -> VecIdx;
        VecIdx -> Retrievers;
        Neo4j -> Retrievers;
        Retrievers -> Groq -> Answer;
        Neo4j -> GDS;
    }
    """
)

st.markdown(
    """
    **The pipeline in five steps:**

    1. **Seed** — search OpenAlex for a topic (e.g. *"diffusion models"*), fetch up to *N* papers
       with their authors, concepts, venues, and the IDs of every work they reference.
    2. **Ingest** — `MERGE` everything into Neo4j. Citation edges create stub `:Paper` nodes for
       referenced works so the graph structure is preserved even when we don't fetch them.
    3. **Embed** — encode each paper's abstract into a 384-dim vector and store it on the node.
       The vector index enables fast semantic search.
    4. **Retrieve** — for a given question, hit the vector index to find candidate papers, then
       **expand through the graph** to collect concepts, co-authors, and cited works as context.
    5. **Synthesize** — hand the enriched context to Groq's LLM to produce a grounded answer,
       always linked back to verifiable OpenAlex IDs.
    """
)

st.divider()

# ────────────────────────── COMMAND DIRECTORY ──────────────────────────
st.subheader("Pages in this app")

st.markdown(
    "Each page below corresponds to one CLI command. Use the **left sidebar** to navigate."
)

commands = [
    ("🌱 Seed", "Ingest papers from an OpenAlex topic search and embed their abstracts.",
     "`research seed \"<topic>\" --max-papers N --depth D`"),
    ("📈 Stats", "See current graph contents — node/relationship counts and embedding coverage.",
     "`research stats`"),
    ("🔍 Retrieve", "Retrieve papers with one of four retrievers (vector, graph, hybrid, cypher). No LLM synthesis.",
     "`research retrieve \"<q>\" --mode vector|graph|hybrid|cypher`"),
    ("💬 Ask", "Full GraphRAG Q&A: vector+graph retrieval, then Groq synthesizes an answer with OpenAlex sources.",
     "`research ask \"<q>\" --top-k N`"),
    ("📚 Reading List", "Seminal papers for a topic, ordered chronologically, each with an LLM-written rationale.",
     "`research reading-list \"<topic>\" --level <lvl>`"),
    ("🧬 Lineage", "Find the shortest citation path between two papers and narrate the intellectual evolution.",
     "`research lineage --from W... --to W...`"),
    ("🔀 Gaps", "Analyze the intersection of two concepts — well-studied, or an underexplored gap?",
     "`research gaps \"<concept_a>\" \"<concept_b>\"`"),
    ("📊 Analyze", "Run GDS graph algorithms over the citation network: PageRank, Louvain, betweenness.",
     "`research analyze --algo pagerank|louvain|betweenness`"),
]

for title, desc, cli in commands:
    with st.container(border=True):
        cols = st.columns([1, 2.5, 2])
        with cols[0]:
            st.markdown(f"**{title}**")
        with cols[1]:
            st.markdown(desc)
        with cols[2]:
            st.code(cli, language="bash")

st.divider()

# ─────────────────────────── PREREQS / STATUS ──────────────────────────
with st.expander("Setup checklist before using the pages", expanded=False):
    st.markdown(
        """
        - [x] Python 3.11+, `pip install -e ".[dev]"`
        - [x] `.env` filled in (Neo4j password, Groq `OPENAI_API_KEY`, email)
        - [x] `research init` run once (schema + vector index applied)
        - [x] Neo4j Desktop instance **running** (port 7687 open)
        - [x] GDS plugin installed **on this instance** if you plan to use the Analyze page
        - [x] At least one `research seed` run so there's data to query

        If any of the above isn't done, the pages will show a clear error.
        """
    )

st.caption(
    "Docs: [implementation plan](../docs/implementation-plan.md) · "
    "[curated queries](../docs/queries.md) · "
    "[`neo4j-graphrag`](https://github.com/neo4j/neo4j-graphrag-python)"
)
