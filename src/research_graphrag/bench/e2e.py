import json
import re
from typing import Any

from ..pipelines.research_rag import ask
from ..retrieval.retrievers import get_llm

# The 6 success-criteria questions from docs/implementation-plan.md §11
DEFAULT_QUESTIONS = [
    "Which 5 papers should I read to understand diffusion models?",
    "Trace the evolution of attention mechanisms from 2015 to 2020.",
    "What are the most influential papers in multi-agent reinforcement learning?",
    "Which research communities exist within NLP?",
    "Find underexplored gaps between computer vision and causal inference.",
    "Who are the bridge authors between theoretical and applied ML?",
]

JUDGE_PROMPT = """You are a senior researcher reviewing a GraphRAG system's answer. Rate the answer on these criteria:

- relevance (1-5): Does it directly address the question?
- specificity (1-5): Does it cite concrete papers or authors (by name or ID), or is it generic?
- grounding (1-5): Do the cited sources actually appear to support the claims?
- usefulness (1-5): Would a real researcher find this answer a helpful starting point?

Question:
{question}

Answer:
---
{answer}
---

Retrieved sources (OpenAlex IDs):
{sources}

Return valid JSON: {{"relevance": N, "specificity": N, "grounding": N, "usefulness": N, "notes": "<one sentence>"}}
"""


def _parse(text: str) -> dict[str, Any] | None:
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def run_e2e_suite(questions: list[str] | None = None, top_k: int = 8) -> dict[str, Any]:
    questions = questions or DEFAULT_QUESTIONS
    llm = get_llm()
    results = []

    for q in questions:
        try:
            ans = ask(q, top_k=top_k)
        except Exception as exc:
            results.append({"question": q, "error": f"ask failed: {exc}"})
            continue
        sources_block = ", ".join(s["openalex_id"] for s in ans["sources"]) or "(none)"
        resp = llm.invoke(
            JUDGE_PROMPT.format(question=q, answer=ans["answer"], sources=sources_block)
        )
        parsed = _parse(resp.content) or {}
        results.append({
            "question": q,
            "answer": ans["answer"],
            "sources": [s["openalex_id"] for s in ans["sources"]],
            "scores": {
                k: parsed.get(k)
                for k in ("relevance", "specificity", "grounding", "usefulness")
            },
            "notes": parsed.get("notes"),
        })

    def _avg(field: str) -> float | None:
        vals = [r["scores"].get(field) for r in results if "scores" in r]
        nums = [v for v in vals if isinstance(v, (int, float))]
        return (sum(nums) / len(nums)) if nums else None

    return {
        "per_question": results,
        "averages": {
            "relevance": _avg("relevance"),
            "specificity": _avg("specificity"),
            "grounding": _avg("grounding"),
            "usefulness": _avg("usefulness"),
        },
    }
