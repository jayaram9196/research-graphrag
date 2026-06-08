from collections.abc import AsyncIterator
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..config import get_settings

OPENALEX_BASE = "https://api.openalex.org"
MAX_REFERENCES_PER_PAPER = 100


def _strip_id(url_id: str | None) -> str | None:
    if not url_id:
        return None
    return url_id.rsplit("/", 1)[-1]


def _reconstruct_abstract(inverted: dict[str, list[int]] | None) -> str | None:
    if not inverted:
        return None
    positions: list[tuple[int, str]] = []
    for word, idxs in inverted.items():
        for i in idxs:
            positions.append((i, word))
    positions.sort()
    return " ".join(word for _, word in positions)


def _normalize_work(work: dict[str, Any]) -> dict[str, Any]:
    primary_location = work.get("primary_location") or {}
    source = primary_location.get("source") or {}

    authorships = work.get("authorships") or []
    authors: list[dict[str, Any]] = []
    for a in authorships:
        author = a.get("author") or {}
        if not author.get("id"):
            continue
        institutions = [
            {
                "id": _strip_id(i.get("id")),
                "name": i.get("display_name"),
                "country": i.get("country_code"),
                "type": i.get("type"),
            }
            for i in (a.get("institutions") or [])
            if i.get("id")
        ]
        authors.append(
            {
                "id": _strip_id(author.get("id")),
                "name": author.get("display_name"),
                "orcid": _strip_id(author.get("orcid")) if author.get("orcid") else None,
                "position": a.get("author_position"),
                "institutions": institutions,
            }
        )

    concepts = [
        {
            "id": _strip_id(c.get("id")),
            "name": c.get("display_name"),
            "level": c.get("level"),
            "score": c.get("score"),
            "wikidata_url": c.get("wikidata"),
        }
        for c in (work.get("concepts") or [])
        if c.get("id")
    ]

    referenced = [
        _strip_id(r) for r in (work.get("referenced_works") or [])[:MAX_REFERENCES_PER_PAPER] if r
    ]

    venue = None
    if source.get("id"):
        venue = {
            "id": _strip_id(source.get("id")),
            "name": source.get("display_name"),
            "type": source.get("type"),
            "issn": source.get("issn_l"),
        }

    return {
        "id": _strip_id(work.get("id")),
        "doi": work.get("doi"),
        "title": work.get("title") or work.get("display_name"),
        "abstract": _reconstruct_abstract(work.get("abstract_inverted_index")),
        "year": work.get("publication_year"),
        "citation_count": work.get("cited_by_count", 0),
        "openalex_type": work.get("type"),
        "venue": venue,
        "authors": authors,
        "concepts": concepts,
        "referenced_works": [r for r in referenced if r],
    }


def _normalize_author(author: dict[str, Any]) -> dict[str, Any]:
    summary_stats = author.get("summary_stats") or {}
    return {
        "id": _strip_id(author.get("id")),
        "name": author.get("display_name"),
        "orcid": _strip_id(author.get("orcid")) if author.get("orcid") else None,
        "h_index": summary_stats.get("h_index"),
        "works_count": author.get("works_count"),
    }


class OpenAlexClient:
    """Async client for the OpenAlex API with polite-pool headers and retry."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    @classmethod
    def create(cls) -> "OpenAlexClient":
        settings = get_settings()
        email = settings.openalex_user_agent_email or ""
        headers = {"User-Agent": f"research-graphrag/0.1 ({email})"}
        default_params: dict[str, str] = {}
        if email:
            default_params["mailto"] = email
        client = httpx.AsyncClient(
            base_url=OPENALEX_BASE,
            headers=headers,
            params=default_params,
            timeout=30.0,
        )
        return cls(client)

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "OpenAlexClient":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    @retry(
        reraise=True,
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TransportError)),
    )
    async def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        response = await self._client.get(path, params=params)
        response.raise_for_status()
        return response.json()

    async def search_works(
        self,
        query: str,
        per_page: int = 25,
        max_results: int | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        cursor: str | None = "*"
        yielded = 0
        while cursor:
            payload = await self._get(
                "/works",
                params={"search": query, "per-page": per_page, "cursor": cursor},
            )
            results = payload.get("results") or []
            if not results:
                break
            for work in results:
                yield _normalize_work(work)
                yielded += 1
                if max_results is not None and yielded >= max_results:
                    return
            cursor = (payload.get("meta") or {}).get("next_cursor")

    async def fetch_work(self, openalex_id: str) -> dict[str, Any]:
        payload = await self._get(f"/works/{openalex_id}")
        return _normalize_work(payload)

    async def fetch_author(self, openalex_id: str) -> dict[str, Any]:
        payload = await self._get(f"/authors/{openalex_id}")
        return _normalize_author(payload)
