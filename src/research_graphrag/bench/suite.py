import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .audit import run_graph_audit
from .citation import run_citation_suite
from .community import run_community_coherence
from .e2e import run_e2e_suite
from .faithfulness import run_faithfulness_suite
from .latency import run_latency_suite
from .retrieval import run_retrieval_suite

AVAILABLE = {
    "retrieval": run_retrieval_suite,
    "latency": run_latency_suite,
    "citation": run_citation_suite,
    "audit": run_graph_audit,
    "faithfulness": run_faithfulness_suite,
    "community": run_community_coherence,
    "e2e": run_e2e_suite,
}

REPORTS_DIR = Path("benchmarks/reports")


def run_suite(metrics: list[str] | None = None) -> dict[str, Any]:
    names = metrics or list(AVAILABLE.keys())
    report: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metrics": {},
    }
    for name in names:
        if name not in AVAILABLE:
            report["metrics"][name] = {"error": f"Unknown metric: {name}"}
            continue
        try:
            report["metrics"][name] = AVAILABLE[name]()
        except Exception as exc:
            report["metrics"][name] = {"error": f"{type(exc).__name__}: {exc}"}
    return report


def save_report(report: dict[str, Any], output_dir: Path | None = None) -> Path:
    output_dir = output_dir or REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = report["timestamp"].replace(":", "-")
    path = output_dir / f"{timestamp}.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    latest = output_dir / "latest.json"
    latest.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return path


def summarise(report: dict[str, Any]) -> dict[str, Any]:
    """Extract the top-line numbers for quick display."""
    metrics = report.get("metrics", {})
    summary: dict[str, Any] = {}

    retrieval = metrics.get("retrieval") or {}
    if retrieval.get("results"):
        summary["retrieval_mean_recall"] = {
            r["retriever"]: r["mean_recall"] for r in retrieval["results"]
        }

    audit = metrics.get("audit") or {}
    if "correctness_rate" in audit:
        summary["graph_correctness_rate"] = audit["correctness_rate"]

    citation = metrics.get("citation") or {}
    totals = citation.get("totals") or {}
    if totals.get("sources_verifiability_rate") is not None:
        summary["sources_verifiability_rate"] = totals["sources_verifiability_rate"]
    if totals.get("inline_verifiability_rate") is not None:
        summary["inline_verifiability_rate"] = totals["inline_verifiability_rate"]

    faithfulness = metrics.get("faithfulness") or {}
    if faithfulness.get("mean_faithfulness") is not None:
        summary["faithfulness_mean"] = faithfulness["mean_faithfulness"]

    e2e = metrics.get("e2e") or {}
    if e2e.get("averages"):
        summary["e2e_averages"] = e2e["averages"]

    community = metrics.get("community") or {}
    if community.get("mean_rating") is not None:
        summary["community_coherence_mean"] = community["mean_rating"]

    latency = metrics.get("latency") or {}
    if latency:
        summary["latency_p50"] = {
            k: v.get("p50") for k, v in latency.items() if isinstance(v, dict)
        }

    return summary
