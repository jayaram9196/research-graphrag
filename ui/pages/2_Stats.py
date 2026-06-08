import pandas as pd
import streamlit as st

from research_graphrag.clients.neo4j_client import get_session

from _helpers import require_neo4j

st.set_page_config(page_title="Stats", page_icon="📈", layout="wide")

st.title("📈 Stats")
st.markdown(
    "Current graph contents — node / relationship counts and embedding coverage. "
    "Use this to verify the `research init` schema was applied and to see the scale "
    "of the corpus after seeding."
)

if not require_neo4j():
    st.stop()

if st.button("🔄 Refresh"):
    st.rerun()

with get_session() as session:
    node_rows = session.execute_read(
        lambda tx: [
            {"label": r["label"], "count": r["count"]}
            for r in tx.run(
                """
                MATCH (n)
                RETURN labels(n)[0] AS label, count(*) AS count
                ORDER BY count DESC
                """
            )
        ]
    )
    rel_rows = session.execute_read(
        lambda tx: [
            {"rel_type": r["rel_type"], "count": r["count"]}
            for r in tx.run(
                """
                MATCH ()-[r]->()
                RETURN type(r) AS rel_type, count(*) AS count
                ORDER BY count DESC
                """
            )
        ]
    )
    paper_coverage = session.execute_read(
        lambda tx: tx.run(
            """
            MATCH (p:Paper)
            RETURN count(p) AS total,
                   count(p.title) AS with_title,
                   count(p.abstract) AS with_abstract,
                   count(p.embedding) AS with_embedding
            """
        ).single()
    )

total = paper_coverage["total"] if paper_coverage else 0
with_title = paper_coverage["with_title"] if paper_coverage else 0
with_abstract = paper_coverage["with_abstract"] if paper_coverage else 0
with_embedding = paper_coverage["with_embedding"] if paper_coverage else 0

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total :Paper", total)
m2.metric("With title", with_title, help="Fully ingested (vs citation stubs)")
m3.metric("With abstract", with_abstract)
m4.metric("With embedding", with_embedding)

if total > 0:
    embed_pct = 100 * with_embedding / total
    st.progress(embed_pct / 100, text=f"Embedding coverage: {embed_pct:.1f}% of all :Paper nodes")

st.subheader("Nodes")
if node_rows:
    st.dataframe(pd.DataFrame(node_rows), width="stretch", hide_index=True)
else:
    st.info("Graph is empty. Run the Seed page first.")

st.subheader("Relationships")
if rel_rows:
    st.dataframe(pd.DataFrame(rel_rows), width="stretch", hide_index=True)
