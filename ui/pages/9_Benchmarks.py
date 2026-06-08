import json
from pathlib import Path

import pandas as pd
import streamlit as st

from research_graphrag.bench.suite import AVAILABLE, run_suite, save_report, summarise

from _helpers import require_neo4j

st.set_page_config(page_title="Benchmarks", page_icon="📏", layout="wide")

st.title("📏 Benchmarks")
st.markdown(
    """
    Run quality, correctness, and latency metrics against the current graph and archive
    the results to `benchmarks/reports/`. Use the latest report to spot regressions when
    you tweak the retriever, prompts, or ingestion.

    | Metric | What it measures |
    |--------|------------------|
    | **retrieval** | Recall@K and precision@K on a curated gold set (`benchmarks/gold_set.json`) |
    | **latency** | p50 / p95 over repeated runs of retrieve, ask, pagerank |
    | **citation** | How many OpenAlex IDs the LLM cites actually exist in the graph |
    | **audit** | Sample N papers, re-fetch from OpenAlex live, compare title/year/citations |
    | **faithfulness** | LLM-as-judge: % of claims in an `ask` answer supported by retrieved sources |
    | **community** | LLM-as-judge: coherence of Louvain clusters on a 1-5 scale |
    | **e2e** | LLM-as-judge: relevance, specificity, grounding, usefulness on the plan's 6 demo questions |
    """
)

REPORTS_DIR = Path("benchmarks/reports")
GOLD_PATH = Path("benchmarks/gold_set.json")

# ─────────────────────── RUN ───────────────────────
st.divider()
st.subheader("Run benchmarks")

if not require_neo4j():
    st.stop()

all_metrics = list(AVAILABLE.keys())
with st.form("bench_form"):
    selected = st.multiselect(
        "Metrics",
        all_metrics,
        default=["retrieval", "latency", "citation", "audit"],
        help="LLM-as-judge metrics (faithfulness, community, e2e) are slower and use Groq credits.",
    )
    save = st.checkbox("Save report to disk", value=True)
    submit = st.form_submit_button("Run selected", type="primary")

if submit:
    if not selected:
        st.error("Pick at least one metric.")
        st.stop()
    with st.spinner(f"Running {len(selected)} metric(s)..."):
        report = run_suite(selected)
    if save:
        path = save_report(report)
        st.success(f"Report saved to `{path}`")
    else:
        st.success("Completed (report not persisted).")

    st.subheader("Summary")
    summary = summarise(report)
    if summary:
        st.json(summary)
    st.subheader("Full report")
    st.json(report)
    st.stop()

# ─────────────────────── BROWSE HISTORIC REPORTS ───────────────────────
st.divider()
st.subheader("Historic reports")

report_files = sorted(REPORTS_DIR.glob("*.json"), reverse=True) if REPORTS_DIR.exists() else []
report_files = [p for p in report_files if p.name != "latest.json"]

if not report_files:
    st.info(
        "No reports yet. Run a benchmark above, or from the command line: `research bench`."
    )
    st.stop()

choices = {p.name: p for p in report_files}
selected_name = st.selectbox("Report", list(choices.keys()))
selected_path = choices[selected_name]
report = json.loads(selected_path.read_text(encoding="utf-8"))
summary = summarise(report)

st.caption(f"Timestamp (UTC): {report.get('timestamp')}")

# ── Summary tiles ──
if summary:
    tiles = []
    if "retrieval_mean_recall" in summary:
        for r, score in summary["retrieval_mean_recall"].items():
            tiles.append((f"Retrieval recall — {r}", f"{score:.2%}"))
    if "graph_correctness_rate" in summary:
        tiles.append(("Graph correctness", f"{summary['graph_correctness_rate']:.1%}"))
    if "citation_verifiability_rate" in summary:
        tiles.append(("Citation verifiability", f"{summary['citation_verifiability_rate']:.1%}"))
    if "faithfulness_mean" in summary:
        tiles.append(("Faithfulness (mean)", f"{summary['faithfulness_mean']:.1%}"))
    if "community_coherence_mean" in summary:
        tiles.append(("Community coherence", f"{summary['community_coherence_mean']:.2f} / 5"))
    if "e2e_averages" in summary:
        avgs = summary["e2e_averages"] or {}
        for field in ("relevance", "specificity", "grounding", "usefulness"):
            val = avgs.get(field)
            if val is not None:
                tiles.append((f"E2E {field}", f"{val:.2f} / 5"))

    if tiles:
        n_cols = min(4, len(tiles))
        for i in range(0, len(tiles), n_cols):
            row = tiles[i : i + n_cols]
            cols = st.columns(len(row))
            for (label, value), col in zip(row, cols):
                col.metric(label, value)

# ── Per-metric sections ──
metrics = report.get("metrics") or {}

if "retrieval" in metrics:
    st.subheader("Retrieval")
    res = metrics["retrieval"]
    if "error" in res:
        st.warning(res["error"])
    else:
        st.caption(f"Gold set size: {res['gold_set_size']}, top_k: {res['top_k']}")
        agg = pd.DataFrame([
            {
                "retriever": r["retriever"],
                "mean_recall": r["mean_recall"],
                "mean_precision": r["mean_precision"],
            }
            for r in res.get("results", [])
        ])
        if not agg.empty:
            st.dataframe(agg, use_container_width=True, hide_index=True)
        with st.expander("Per-query details"):
            for r in res.get("results", []):
                st.markdown(f"**{r['retriever']}**")
                st.dataframe(
                    pd.DataFrame(r["per_query"]),
                    use_container_width=True,
                    hide_index=True,
                )

if "latency" in metrics:
    st.subheader("Latency")
    lat = metrics["latency"]
    if isinstance(lat, dict) and "error" not in lat:
        rows = []
        for label, stats in lat.items():
            if isinstance(stats, dict):
                rows.append({"command": label, **stats})
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

if "citation" in metrics:
    st.subheader("Citation verifiability")
    cit = metrics["citation"]
    if "totals" in cit:
        st.json(cit["totals"])
    with st.expander("Per-question detail"):
        st.dataframe(
            pd.DataFrame(cit.get("questions", [])),
            use_container_width=True,
            hide_index=True,
        )

if "audit" in metrics:
    st.subheader("Graph audit vs OpenAlex")
    aud = metrics["audit"]
    if "error" in aud:
        st.warning(aud["error"])
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Correctness", f"{aud['correctness_rate']:.1%}")
        col2.metric("Title match", f"{aud['title_match_rate']:.1%}")
        col3.metric("Year match", f"{aud['year_match_rate']:.1%}")
        with st.expander("Per-paper findings"):
            st.dataframe(
                pd.DataFrame(aud.get("findings", [])),
                use_container_width=True,
                hide_index=True,
            )

if "faithfulness" in metrics:
    st.subheader("Faithfulness (LLM-as-judge)")
    f = metrics["faithfulness"]
    if f.get("mean_faithfulness") is not None:
        st.metric("Mean faithfulness", f"{f['mean_faithfulness']:.1%}")
    with st.expander("Per-question claim verdicts"):
        st.json(f.get("per_question", []))

if "community" in metrics:
    st.subheader("Community coherence (LLM-as-judge)")
    c = metrics["community"]
    if c.get("mean_rating") is not None:
        st.metric("Mean rating", f"{c['mean_rating']:.2f} / 5")
    st.dataframe(
        pd.DataFrame(c.get("communities", [])),
        use_container_width=True,
        hide_index=True,
    )

if "e2e" in metrics:
    st.subheader("End-to-end quality (LLM-as-judge)")
    e = metrics["e2e"]
    if e.get("averages"):
        st.json(e["averages"])
    with st.expander("Per-question results"):
        st.json(e.get("per_question", []))

st.divider()
st.caption(
    f"Gold set source: `{GOLD_PATH}` — edit this file to customize recall@K queries for your corpus."
)
