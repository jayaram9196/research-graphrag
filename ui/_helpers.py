"""Shared helpers for the Streamlit UI."""

from __future__ import annotations

import streamlit as st

from research_graphrag.clients.neo4j_client import get_driver


def require_neo4j() -> bool:
    """Attempt to connect; return True on success, render an error otherwise."""
    try:
        driver = get_driver()
        driver.verify_connectivity()
        return True
    except Exception as exc:
        st.error(
            "Couldn't connect to Neo4j. Start your Neo4j Desktop instance and "
            "check the URI/credentials in `.env`."
        )
        st.caption(f"Details: {exc}")
        return False


def paper_link(paper_id: str | None) -> str:
    if not paper_id:
        return ""
    return f"https://openalex.org/{paper_id}"


def openalex_markdown_link(paper_id: str | None, label: str | None = None) -> str:
    if not paper_id:
        return label or ""
    return f"[{label or paper_id}]({paper_link(paper_id)})"
