# Research Paper GraphRAG — Python

A working GraphRAG system for exploring research paper citation networks. Ingests papers from [OpenAlex](https://openalex.org) into Neo4j (citations, authors, concepts, institutions, venues), then answers questions with graph-grounded retrieval and LLM synthesis.

Built on Neo4j's official [`neo4j-graphrag`](https://github.com/neo4j/neo4j-graphrag-python) library.

## What it does

- **Ingest** — BFS over OpenAlex search + citation graph; idempotent `MERGE` writes
- **Embed** — batch-encode abstracts (fastembed / SentenceTransformers, 384-dim)
- **Retrieve** — 4 retrievers: vector, hybrid, vector+graph (the key one), text-to-cypher
- **Synthesize** — 4 LLM-backed pipelines: ask, reading-list, lineage, gaps
- **Analyze** — GDS graph algorithms: PageRank, Louvain, betweenness

## Stack

| Layer | Tool |
|-------|------|
| Graph DB | Neo4j Desktop (local, 5.x) + GDS plugin |
| GraphRAG | `neo4j-graphrag` |
| LLM | Groq via OpenAI-compatible API (free tier) |
| Embeddings | `fastembed` with `sentence-transformers/all-MiniLM-L6-v2` |
| Data | OpenAlex (free, ~240M works) |
| CLI | `typer` + `rich` |

## Prerequisites

1. **Neo4j Desktop 2.x** installed, with an instance running and the GDS plugin enabled for that instance
2. **Python 3.11+**
3. **Groq API key** — sign up free at [console.groq.com](https://console.groq.com)

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows; on macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env
# Edit .env — fill in NEO4J_PASSWORD, OPENAI_API_KEY (Groq key), OPENALEX_USER_AGENT_EMAIL
```

Then apply the schema:

```bash
research init
```

Creates 5 uniqueness constraints, `paper_year` / `paper_title` indexes, a fulltext index on `[title, abstract]`, and the `paper_abstracts` vector index at 384 dimensions.

## Quick demo

```bash
research seed "diffusion models" --max-papers 200
research ask "Trace the evolution from DDPM to Stable Diffusion" --top-k 8
research analyze --algo louvain --limit 5 --min-size 10
```

## Web UI (Streamlit)

Prefer clicking over typing? There's a Streamlit app with one page per command.

```bash
streamlit run ui/app.py
```

Opens at http://localhost:8501 with:
- **Home** — project overview, architecture diagram, command directory
- **Seed / Stats** — ingest data, inspect the graph
- **Retrieve / Ask** — query with either raw retrieval or full GraphRAG synthesis
- **Reading List / Lineage / Gaps** — the three purpose-built pipelines
- **Analyze** — PageRank / Louvain / betweenness (requires GDS)

## CLI reference

| Command | What it does |
|---------|--------------|
| `research init` | Apply schema (constraints, indexes, vector index) |
| `research seed "<topic>" [-n N] [-d D]` | Search OpenAlex, BFS ingest up to `N` papers at depth `D`, then embed |
| `research embed [--limit N]` | Embed any papers that have an abstract but no embedding |
| `research stats` | Node / relationship counts, embedding coverage |
| `research retrieve "<q>" --mode vector\|graph\|hybrid\|cypher [-k N]` | Retrieve papers (no LLM synthesis) |
| `research ask "<q>" [-k N]` | GraphRAG Q&A: vector+graph retrieval → LLM answer + sources |
| `research reading-list "<topic>" [-l level] [-n N]` | Seminal papers for a topic, chronological, with rationale |
| `research lineage --from W... --to W... [--max-depth D]` | Narrate the citation path between two papers |
| `research gaps "<concept_a>" "<concept_b>"` | Analyze the intersection (or gap) between two concepts |
| `research analyze --algo pagerank\|louvain\|betweenness [-c concept] [-n N]` | Run a GDS algorithm on the citation graph |
| `research version` | Print CLI version |

## Architecture

```
src/research_graphrag/
├── config.py                 # pydantic-settings
├── cli.py                    # typer commands
├── clients/
│   ├── neo4j_client.py       # driver singleton + session context mgr
│   └── openalex.py           # async httpx client (polite pool, tenacity retry)
├── schema/
│   ├── models.py             # Pydantic Paper/Author/Institution/Venue/Concept
│   └── cypher.py             # idempotent DDL
├── ingest/
│   ├── seed.py               # BFS ingestion, MERGE-everywhere
│   └── embed.py              # batch embedding via fastembed or sentence-transformers
├── retrieval/
│   ├── embeddings.py         # QueryEmbedder (satisfies neo4j-graphrag Embedder)
│   └── retrievers.py         # 4 retriever factories + Groq LLM wiring
├── pipelines/
│   ├── research_rag.py       # general Q&A
│   ├── reading_list.py       # concept-scoped, chronological, LLM rationale
│   ├── lineage.py            # shortestPath + LLM narrative
│   └── gaps.py               # concept intersection + LLM interpretation
└── algorithms/
    ├── _gds.py               # in-memory projection helper
    ├── pagerank.py
    ├── communities.py        # Louvain
    └── centrality.py         # betweenness
```

## Design notes

- **LLM choice**: Groq is OpenAI-API-compatible, so `neo4j-graphrag`'s `OpenAILLM` is reused with `base_url=https://api.groq.com/openai/v1`. Default model is `llama-3.3-70b-versatile`.
- **Embedding model**: 384-dim MiniLM via `fastembed` (ONNX, downloaded from Qdrant's CDN — avoids HuggingFace connectivity issues on some networks). Same model works via `sentence-transformers` if you prefer.
- **Citation stubs**: BFS at `--max-papers N` fully ingests `N` unique papers. Their cited references become `:Paper` nodes with only an `id` (no title/abstract). This preserves the citation graph while capping work. Re-running with a larger cap (or via `--depth`) fills them in — `MERGE` is idempotent.
- **Dedup**: every write is a `MERGE` on OpenAlex ID. Re-running the same seed updates rather than duplicates.

## Troubleshooting

- **HuggingFace connection reset on first embedding run** — switch `EMBEDDING_PROVIDER=fastembed` in `.env` (already the default). fastembed downloads from Qdrant's CDN instead.
- **`ServiceUnavailable` / `10061` connecting to Neo4j** — Neo4j Desktop 2.x reuses port 7687 but may not rebind immediately after a restart. Check `Test-NetConnection -ComputerName 127.0.0.1 -Port 7687`. Prefer `bolt://127.0.0.1:7687` over `neo4j://` to skip routing discovery.
- **`gds.version() not found`** — GDS is installed per-instance in Neo4j Desktop 2.x. Open the instance, install GDS from its Plugins tab, then restart.
- **`research init` fails with vector index dimension mismatch** — you switched embedding models. Drop the old vector index manually in Neo4j Browser (`DROP INDEX paper_abstracts`) and re-run `research init`.

## Further reading

- Implementation plan: [`docs/implementation-plan.md`](docs/implementation-plan.md)
- Curated Cypher queries: [`docs/queries.md`](docs/queries.md)
