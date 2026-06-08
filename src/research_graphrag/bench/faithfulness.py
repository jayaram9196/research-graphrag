import json
import re
from typing import Any

from ..pipelines.research_rag import ask
from ..retrieval.retrievers import get_llm

CLAIM_EXTRACTION_PROMPT = """Extract every standalone factual claim from the answer below.
A claim is a single factual statement that can be independently verified (e.g. "Paper X introduced technique Y in 2020", "Method A is computationally more expensive than Method B").

Return valid JSON with this exact shape:
{"claims": ["claim 1", "claim 2", ...]}

Answer:
---
{answer}
---
"""

JUDGE_PROMPT = """You are a careful reviewer. Determine whether each claim is supported by the provided source snippets. A claim is "supported" only if it is directly stated or strongly implied by the sources.

Sources:
---
{sources}
---

Claims:
{numbered_claims}

Return valid JSON with this exact shape:
{{"verdicts": [{{"claim_index": 1, "verdict": "supported|unsupported|partial", "reason": "..."}}, ...]}}
"""


def _parse_json_obj(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _extract_claims(llm, answer: str) -> list[str]:
    resp = llm.invoke(CLAIM_EXTRACTION_PROMPT.format(answer=answer))
    parsed = _parse_json_obj(resp.content) or {}
    claims = parsed.get("claims") or []
    return [str(c).strip() for c in claims if str(c).strip()]


def _judge_claims(llm, claims: list[str], sources_text: str) -> list[dict[str, Any]]:
    numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims))
    resp = llm.invoke(JUDGE_PROMPT.format(sources=sources_text, numbered_claims=numbered))
    parsed = _parse_json_obj(resp.content) or {}
    return parsed.get("verdicts") or []


def score_answer_faithfulness(
    question: str,
    top_k: int = 6,
) -> dict[str, Any]:
    result = ask(question, top_k=top_k)
    answer = result["answer"]
    context_items = []
    if result.get("sources"):
        for s in result["sources"]:
            context_items.append(f"[{s['openalex_id']}]")
    sources_text = result.get("context_text") or "\n".join(context_items) or "(no retrieved context)"

    llm = get_llm()
    claims = _extract_claims(llm, answer)
    if not claims:
        return {
            "question": question,
            "claims": [],
            "verdicts": [],
            "faithfulness_rate": 0.0,
            "note": "No claims extracted.",
        }
    verdicts = _judge_claims(llm, claims, sources_text)
    supported = sum(1 for v in verdicts if v.get("verdict") == "supported")
    partial = sum(1 for v in verdicts if v.get("verdict") == "partial")
    return {
        "question": question,
        "claims": claims,
        "verdicts": verdicts,
        "n_claims": len(claims),
        "supported": supported,
        "partial": partial,
        "faithfulness_rate": (supported + 0.5 * partial) / len(claims) if claims else 0.0,
    }


def run_faithfulness_suite(questions: list[str] | None = None, top_k: int = 6) -> dict[str, Any]:
    questions = questions or [
        "What is classifier-free guidance and why does it matter?",
        "What are the foundational deep learning diffusion models for image generation?",
    ]
    per_question = [score_answer_faithfulness(q, top_k=top_k) for q in questions]
    rates = [r["faithfulness_rate"] for r in per_question]
    return {
        "per_question": per_question,
        "mean_faithfulness": sum(rates) / len(rates) if rates else 0.0,
    }
