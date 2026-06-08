from collections import deque
from typing import Any

from rich.console import Console

from ..clients.neo4j_client import get_session
from ..clients.openalex import OpenAlexClient

console = Console()

UPSERT_PAPERS_CYPHER = """
UNWIND $papers AS p
MERGE (paper:Paper {id: p.id})
  ON CREATE SET paper.created_at = datetime()
SET paper.doi = coalesce(p.doi, paper.doi),
    paper.title = coalesce(p.title, paper.title),
    paper.abstract = coalesce(p.abstract, paper.abstract),
    paper.year = coalesce(p.year, paper.year),
    paper.citation_count = coalesce(p.citation_count, paper.citation_count),
    paper.openalex_type = coalesce(p.openalex_type, paper.openalex_type)

FOREACH (_ IN CASE WHEN p.venue IS NULL THEN [] ELSE [1] END |
  MERGE (v:Venue {id: p.venue.id})
    ON CREATE SET v.name = p.venue.name, v.type = p.venue.type, v.issn = p.venue.issn
  MERGE (paper)-[:PUBLISHED_IN]->(v)
)

FOREACH (a IN p.authors |
  MERGE (author:Author {id: a.id})
    ON CREATE SET author.name = a.name, author.orcid = a.orcid
  MERGE (paper)-[r:AUTHORED_BY]->(author)
    SET r.position = a.position
  FOREACH (inst IN a.institutions |
    MERGE (i:Institution {id: inst.id})
      ON CREATE SET i.name = inst.name, i.country = inst.country, i.type = inst.type
    MERGE (author)-[:AFFILIATED_WITH]->(i)
  )
)

FOREACH (c IN p.concepts |
  MERGE (concept:Concept {id: c.id})
    ON CREATE SET concept.name = c.name, concept.level = c.level, concept.wikidata_url = c.wikidata_url
  MERGE (paper)-[r:ABOUT]->(concept)
    SET r.score = c.score
)

FOREACH (ref_id IN p.referenced_works |
  MERGE (cited:Paper {id: ref_id})
  MERGE (paper)-[:CITES]->(cited)
)
"""

WRITE_BATCH_SIZE = 25


def _write_papers(papers: list[dict[str, Any]]) -> None:
    if not papers:
        return
    with get_session() as session:
        session.execute_write(
            lambda tx: tx.run(UPSERT_PAPERS_CYPHER, papers=papers).consume()
        )


async def seed_topic(topic: str, max_papers: int, depth: int = 1) -> dict[str, Any]:
    """BFS ingest starting from an OpenAlex topic search.

    Layer 0 = top search hits. Layers 1..depth = their referenced_works.
    All writes use MERGE on OpenAlex ID, so re-running the same seed is idempotent.
    """
    seen: set[str] = set()
    queue: deque[tuple[str, int]] = deque()
    layer_counts: dict[int, int] = {}
    batch: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal batch
        if batch:
            _write_papers(batch)
            batch = []

    async with OpenAlexClient.create() as client:
        console.print(f"[bold]Searching OpenAlex:[/bold] {topic!r} (cap: {max_papers} papers)")

        async for work in client.search_works(topic, per_page=25, max_results=max_papers):
            if not work["id"] or work["id"] in seen or len(seen) >= max_papers:
                if len(seen) >= max_papers:
                    break
                continue
            seen.add(work["id"])
            batch.append(work)
            layer_counts[0] = layer_counts.get(0, 0) + 1
            if depth >= 1:
                for ref in work["referenced_works"]:
                    if ref and ref not in seen:
                        queue.append((ref, 1))
            if len(batch) >= WRITE_BATCH_SIZE:
                flush()
        flush()
        console.print(f"  Layer 0: {layer_counts.get(0, 0)} papers")

        while queue and len(seen) < max_papers:
            paper_id, current_depth = queue.popleft()
            if paper_id in seen or current_depth > depth:
                continue
            seen.add(paper_id)
            try:
                work = await client.fetch_work(paper_id)
            except Exception as exc:
                console.print(f"[yellow]  skip {paper_id}: {exc}[/yellow]")
                continue
            batch.append(work)
            layer_counts[current_depth] = layer_counts.get(current_depth, 0) + 1
            if current_depth < depth:
                for ref in work["referenced_works"]:
                    if ref and ref not in seen:
                        queue.append((ref, current_depth + 1))
            if len(batch) >= WRITE_BATCH_SIZE:
                flush()
        flush()

        for d in sorted(layer_counts):
            if d > 0:
                console.print(f"  Layer {d}: {layer_counts[d]} papers")

    return {"total": len(seen), "layer_counts": layer_counts}
