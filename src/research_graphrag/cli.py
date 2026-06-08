import asyncio

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .clients.neo4j_client import close_driver, get_session
from .config import get_settings
from .ingest.embed import embed_pending
from .ingest.seed import seed_topic
from .schema.cypher import apply_schema

app = typer.Typer(
    help="Research paper GraphRAG CLI",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


@app.command()
def version() -> None:
    """Print the CLI version."""
    console.print(f"research-graphrag {__version__}")


@app.command()
def init() -> None:
    """Apply Neo4j schema: constraints, indexes, and the vector index."""
    settings = get_settings()
    console.print(f"[bold]Connecting to[/bold] {settings.neo4j_uri} (db={settings.neo4j_database})")
    try:
        apply_schema()
        console.print("[green]Schema applied.[/green]")

        with get_session() as session:
            constraints = [r["name"] for r in session.run("SHOW CONSTRAINTS YIELD name")]
            indexes = [
                (r["name"], r["type"])
                for r in session.run("SHOW INDEXES YIELD name, type")
            ]

        table = Table(title="Schema", show_header=True, header_style="bold cyan")
        table.add_column("Kind")
        table.add_column("Name")
        table.add_column("Type")
        for name in constraints:
            table.add_row("constraint", name, "-")
        for name, idx_type in indexes:
            table.add_row("index", name, idx_type)
        console.print(table)

        console.print(
            f"[dim]Vector index dimensions: {settings.embedding_dimensions} "
            f"(from embedding_model={settings.embedding_model})[/dim]"
        )
    finally:
        close_driver()


@app.command()
def seed(
    topic: str = typer.Argument(..., help="Free-text topic to search on OpenAlex"),
    max_papers: int = typer.Option(
        None, "--max-papers", "-n", help="Cap on total unique papers (default from .env)"
    ),
    depth: int = typer.Option(
        None, "--depth", "-d", help="Citation BFS depth (default from .env)"
    ),
    skip_embed: bool = typer.Option(
        False, "--skip-embed", help="Skip embedding abstracts after ingest"
    ),
) -> None:
    """Seed the graph by searching OpenAlex and ingesting papers + their citations."""
    settings = get_settings()
    cap = max_papers if max_papers is not None else settings.max_papers_per_seed
    d = depth if depth is not None else settings.max_citation_depth

    try:
        stats = asyncio.run(seed_topic(topic, max_papers=cap, depth=d))
        console.print(f"[green]Ingested {stats['total']} papers.[/green]")

        if not skip_embed:
            console.print()
            embed_stats = embed_pending()
            console.print(
                f"[green]Embedded {embed_stats['embedded']} abstracts.[/green]"
            )
    finally:
        close_driver()


@app.command()
def embed(
    limit: int = typer.Option(None, "--limit", help="Stop after N embeddings"),
) -> None:
    """Embed any papers that have an abstract but no embedding yet."""
    try:
        stats = embed_pending(limit=limit)
        console.print(f"[green]Embedded {stats['embedded']} abstracts.[/green]")
    finally:
        close_driver()


@app.command()
def retrieve(
    question: str = typer.Argument(..., help="Natural-language question"),
    mode: str = typer.Option(
        "graph",
        "--mode",
        "-m",
        help="vector | graph | hybrid | cypher",
    ),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Results to return (ignored for cypher)"),
) -> None:
    """Retrieve relevant papers using the chosen retriever."""
    from .retrieval.retrievers import (
        make_hybrid_retriever,
        make_text2cypher_retriever,
        make_vector_cypher_retriever,
        make_vector_retriever,
    )

    factories = {
        "vector": make_vector_retriever,
        "graph": make_vector_cypher_retriever,
        "hybrid": make_hybrid_retriever,
        "cypher": make_text2cypher_retriever,
    }
    if mode not in factories:
        console.print(f"[red]Unknown mode: {mode}. Use one of: {', '.join(factories)}[/red]")
        raise typer.Exit(code=1)

    try:
        retriever = factories[mode]()
        console.print(f"[bold]Mode:[/bold] {mode}    [bold]Query:[/bold] {question}\n")

        if mode == "cypher":
            result = retriever.search(query_text=question)
        else:
            result = retriever.search(query_text=question, top_k=top_k)

        items = result.items[:top_k]
        if not items:
            console.print("[yellow]No results.[/yellow]")
            return

        if mode == "cypher" and len(result.items) > top_k:
            console.print(
                f"[dim]Generated Cypher returned {len(result.items)} rows; "
                f"showing first {top_k}.[/dim]\n"
            )

        for i, item in enumerate(items, 1):
            console.print(f"[bold cyan]{i}.[/bold cyan] {item.content}")
            meta = item.metadata or {}
            openalex_id = meta.get("openalex_id")
            if openalex_id:
                console.print(f"   [dim]https://openalex.org/{openalex_id}[/dim]")
            console.print()
    finally:
        close_driver()


@app.command()
def ask(
    question: str = typer.Argument(..., help="Free-form research question"),
    top_k: int = typer.Option(5, "--top-k", "-k"),
) -> None:
    """Answer a question using GraphRAG (vector+graph retrieval, then LLM)."""
    from .pipelines.research_rag import ask as ask_impl

    try:
        result = ask_impl(question, top_k=top_k)
        console.print(f"[bold]Answer:[/bold]\n{result['answer']}\n")
        if result["sources"]:
            console.print("[bold]Sources:[/bold]")
            for s in result["sources"]:
                console.print(f"  https://openalex.org/{s['openalex_id']}")
    finally:
        close_driver()


@app.command("reading-list")
def reading_list_cmd(
    topic: str = typer.Argument(..., help="Topic, e.g. 'diffusion models'"),
    level: str = typer.Option("intermediate", "--level", "-l", help="beginner | intermediate | advanced"),
    max_items: int = typer.Option(7, "--max-items", "-n"),
) -> None:
    """Curated reading list on a topic, ordered chronologically with rationale."""
    from .pipelines.reading_list import reading_list

    try:
        result = reading_list(topic, level=level, max_items=max_items)
        console.print(f"[bold]Reading list for[/bold] {topic!r} [bold]({level}):[/bold]\n")
        console.print(result["answer"])
        if result["papers"]:
            console.print("\n[bold]OpenAlex links:[/bold]")
            for p in result["papers"]:
                console.print(f"  https://openalex.org/{p['id']}  — {p['title']}")
    finally:
        close_driver()


@app.command()
def lineage(
    from_id: str = typer.Option(..., "--from", help="OpenAlex ID of source paper (e.g. W123...)"),
    to_id: str = typer.Option(..., "--to", help="OpenAlex ID of target paper"),
    max_depth: int = typer.Option(10, "--max-depth"),
) -> None:
    """Trace the citation path between two papers and narrate the evolution."""
    from .pipelines.lineage import trace_lineage

    try:
        result = trace_lineage(from_id, to_id, max_depth=max_depth)
        if result["papers"]:
            console.print(f"[bold]Path ({len(result['papers'])} papers):[/bold]")
            for p in result["papers"]:
                console.print(
                    f"  [{p['id']}] {p['title']} ({p.get('year')})"
                )
            console.print()
        console.print(f"[bold]Narrative:[/bold]\n{result['answer']}")
    finally:
        close_driver()


@app.command()
def gaps(
    concept_a: str = typer.Argument(..., help="First concept, e.g. 'NLP'"),
    concept_b: str = typer.Argument(..., help="Second concept, e.g. 'robotics'"),
) -> None:
    """Find underexplored intersections between two concepts."""
    from .pipelines.gaps import find_gaps

    try:
        result = find_gaps(concept_a, concept_b)
        console.print(f"[bold]Matched A:[/bold] {result['matched_a']}")
        console.print(f"[bold]Matched B:[/bold] {result['matched_b']}")
        console.print(f"[bold]Intersection:[/bold] {len(result['papers'])} papers\n")
        console.print(result["answer"])
    finally:
        close_driver()


@app.command()
def bench(
    metrics: str = typer.Option(
        "all",
        "--metrics",
        "-m",
        help="Comma-separated: retrieval,latency,citation,audit,faithfulness,community,e2e. Or 'all'.",
    ),
    save: bool = typer.Option(True, "--save/--no-save", help="Write report to benchmarks/reports/"),
) -> None:
    """Run benchmark metrics and print a summary."""
    from .bench.suite import AVAILABLE, run_suite, save_report, summarise

    names = list(AVAILABLE.keys()) if metrics == "all" else [m.strip() for m in metrics.split(",") if m.strip()]
    bad = [n for n in names if n not in AVAILABLE]
    if bad:
        console.print(f"[red]Unknown metrics: {bad}. Use: {list(AVAILABLE)}[/red]")
        raise typer.Exit(code=1)

    try:
        console.print(f"[bold]Running metrics:[/bold] {names}\n")
        report = run_suite(names)
        summary = summarise(report)

        console.print("[bold]Summary[/bold]")
        if not summary:
            console.print("  [yellow](no numeric summary available)[/yellow]")
        for k, v in summary.items():
            console.print(f"  {k}: {v}")

        if save:
            path = save_report(report)
            console.print(f"\n[dim]Report saved to {path}[/dim]")
            console.print("[dim]Open the Benchmarks page in the Streamlit UI to browse.[/dim]")
    finally:
        close_driver()


@app.command()
def analyze(
    algo: str = typer.Option("pagerank", "--algo", "-a", help="pagerank | louvain | betweenness"),
    concept: str = typer.Option(None, "--concept", "-c", help="Filter by concept substring (pagerank/betweenness)"),
    limit: int = typer.Option(15, "--limit", "-n"),
    min_size: int = typer.Option(3, "--min-size", help="Louvain: minimum community size"),
) -> None:
    """Run a GDS graph algorithm over the citation network."""
    from .algorithms.centrality import run_betweenness
    from .algorithms.communities import detect_communities
    from .algorithms.pagerank import run_pagerank

    try:
        if algo == "pagerank":
            console.print(
                f"[bold]PageRank[/bold]"
                + (f" (concept filter: {concept!r})" if concept else "")
            )
            rows = run_pagerank(concept=concept, limit=limit)
            if not rows:
                console.print("[yellow]No results.[/yellow]")
                return
            table = Table(header_style="bold cyan")
            table.add_column("#", justify="right")
            table.add_column("score", justify="right")
            table.add_column("year", justify="right")
            table.add_column("cites", justify="right")
            table.add_column("title")
            for i, r in enumerate(rows, 1):
                table.add_row(
                    str(i),
                    f"{r['score']:.4f}",
                    str(r.get("year") or "?"),
                    str(r.get("citation_count") or 0),
                    (r["title"] or "")[:80],
                )
            console.print(table)
            for r in rows:
                console.print(f"  [dim]https://openalex.org/{r['id']}[/dim]")

        elif algo == "louvain":
            console.print("[bold]Louvain community detection[/bold]")
            rows = detect_communities(limit=limit, min_size=min_size)
            if not rows:
                console.print("[yellow]No communities above min-size.[/yellow]")
                return
            for r in rows:
                console.print(
                    f"\n[bold cyan]Community {r['communityId']}[/bold cyan] — {r['size']} papers"
                )
                for title, pid in zip(r["sample_titles"], r["sample_ids"]):
                    console.print(f"  [{pid}] {title}")

        elif algo == "betweenness":
            console.print(
                f"[bold]Betweenness centrality (bridge papers)[/bold]"
                + (f" (concept filter: {concept!r})" if concept else "")
            )
            rows = run_betweenness(concept=concept, limit=limit)
            if not rows:
                console.print("[yellow]No results.[/yellow]")
                return
            table = Table(header_style="bold cyan")
            table.add_column("#", justify="right")
            table.add_column("score", justify="right")
            table.add_column("year", justify="right")
            table.add_column("cites", justify="right")
            table.add_column("title")
            for i, r in enumerate(rows, 1):
                table.add_row(
                    str(i),
                    f"{r['score']:.2f}",
                    str(r.get("year") or "?"),
                    str(r.get("citation_count") or 0),
                    (r["title"] or "")[:80],
                )
            console.print(table)
            for r in rows:
                console.print(f"  [dim]https://openalex.org/{r['id']}[/dim]")

        else:
            console.print(
                f"[red]Unknown algo: {algo}. Use: pagerank | louvain | betweenness[/red]"
            )
            raise typer.Exit(code=1)
    finally:
        close_driver()


@app.command()
def stats() -> None:
    """Print node and relationship counts."""
    try:
        with get_session() as session:
            rows = list(
                session.run(
                    """
                    MATCH (n)
                    RETURN labels(n)[0] AS kind, count(*) AS cnt
                    ORDER BY cnt DESC
                    """
                )
            )
            rel_rows = list(
                session.run(
                    "MATCH ()-[r]->() RETURN type(r) AS kind, count(*) AS cnt ORDER BY cnt DESC"
                )
            )
            embedded = session.run(
                "MATCH (p:Paper) WHERE p.embedding IS NOT NULL RETURN count(p) AS c"
            ).single()["c"]
            total_papers = session.run("MATCH (p:Paper) RETURN count(p) AS c").single()["c"]

        table = Table(title="Graph contents", header_style="bold cyan")
        table.add_column("Kind")
        table.add_column("Name")
        table.add_column("Count", justify="right")
        for r in rows:
            table.add_row("node", r["kind"] or "(unlabelled)", str(r["cnt"]))
        for r in rel_rows:
            table.add_row("rel", r["kind"], str(r["cnt"]))
        table.add_row("paper", "embedded / total", f"{embedded} / {total_papers}")
        console.print(table)
    finally:
        close_driver()


if __name__ == "__main__":
    app()
