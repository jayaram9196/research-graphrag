import asyncio
import random
from typing import Any

from ..clients.neo4j_client import get_session
from ..clients.openalex import OpenAlexClient


def _sample_titled_papers(limit: int) -> list[dict[str, Any]]:
    with get_session() as session:
        return session.execute_read(
            lambda tx: [
                dict(r)
                for r in tx.run(
                    """
                    MATCH (p:Paper)
                    WHERE p.title IS NOT NULL
                    WITH p, rand() AS r
                    ORDER BY r
                    LIMIT $limit
                    RETURN p.id AS id,
                           p.title AS title,
                           p.year AS year,
                           p.citation_count AS citation_count
                    """,
                    limit=limit,
                )
            ]
        )


async def _audit_sample(sample: list[dict[str, Any]], citation_tol: int = 200) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    async with OpenAlexClient.create() as client:
        for row in sample:
            try:
                live = await client.fetch_work(row["id"])
            except Exception as exc:
                findings.append({
                    "id": row["id"],
                    "ok": False,
                    "error": f"fetch failed: {exc}",
                })
                continue
            title_match = (row["title"] or "").strip() == (live["title"] or "").strip()
            year_match = row.get("year") == live.get("year")
            live_citations = live.get("citation_count") or 0
            our_citations = row.get("citation_count") or 0
            citation_drift = live_citations - our_citations
            citation_close = abs(citation_drift) <= citation_tol
            findings.append({
                "id": row["id"],
                "title_match": title_match,
                "year_match": year_match,
                "our_citations": our_citations,
                "live_citations": live_citations,
                "citation_drift": citation_drift,
                "citation_close": citation_close,
                "ok": title_match and year_match,
            })
    return findings


def run_graph_audit(sample_size: int = 10) -> dict[str, Any]:
    sample = _sample_titled_papers(sample_size)
    if not sample:
        return {"error": "No titled papers to sample.", "findings": []}
    findings = asyncio.run(_audit_sample(sample))
    ok_count = sum(1 for f in findings if f.get("ok"))
    return {
        "sample_size": len(findings),
        "correctness_rate": ok_count / len(findings),
        "title_match_rate": sum(1 for f in findings if f.get("title_match")) / len(findings),
        "year_match_rate": sum(1 for f in findings if f.get("year_match")) / len(findings),
        "findings": findings,
    }
