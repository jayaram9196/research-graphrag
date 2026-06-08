import json
import re
from typing import Any

from ..algorithms.communities import detect_communities
from ..retrieval.retrievers import get_llm

JUDGE_PROMPT = """You are evaluating whether the papers below form a coherent research community.
A coherent community means they share the same core research area, not just a common keyword.

Papers (titles only):
{titles}

Rate coherence on a 1-5 scale:
1 = completely unrelated
2 = share a term but cover different fields
3 = related but diverse subtopics
4 = coherent subfield
5 = tightly coherent specialization

Return valid JSON: {{"rating": <1-5>, "label": "<short label, e.g. 'ML image synthesis'>", "reason": "<one sentence>"}}
"""


def _parse(text: str) -> dict[str, Any] | None:
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def run_community_coherence(limit: int = 5, min_size: int = 5) -> dict[str, Any]:
    """Run Louvain, then ask the LLM to rate each top community's coherence."""
    try:
        communities = detect_communities(limit=limit, min_size=min_size)
    except Exception as exc:
        return {"error": f"Louvain failed (GDS not available?): {exc}", "communities": []}

    if not communities:
        return {"error": "No communities to rate.", "communities": []}

    llm = get_llm()
    ratings: list[dict[str, Any]] = []
    for comm in communities:
        titles_block = "\n".join(f"- {t}" for t in comm["sample_titles"])
        resp = llm.invoke(JUDGE_PROMPT.format(titles=titles_block))
        parsed = _parse(resp.content) or {}
        ratings.append({
            "communityId": comm["communityId"],
            "size": comm["size"],
            "sample_titles": comm["sample_titles"],
            "rating": parsed.get("rating"),
            "label": parsed.get("label"),
            "reason": parsed.get("reason"),
        })

    numeric = [r["rating"] for r in ratings if isinstance(r.get("rating"), (int, float))]
    return {
        "communities": ratings,
        "mean_rating": sum(numeric) / len(numeric) if numeric else None,
    }
