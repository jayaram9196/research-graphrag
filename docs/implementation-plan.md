# Research Paper GraphRAG — Python Implementation Plan

A production-oriented GraphRAG system for research papers, built on Python with Neo4j's official `neo4j-graphrag` library. Replaces the previous TypeScript/Genkit implementation.

---

## 1. Why this rewrite

The TypeScript version on Genkit works but had to hand-roll everything beyond vector search. `neo4j-graphrag` is Neo4j's first-party library and gives us, out of the box:

- **Retrievers**: `VectorRetriever`, `HybridRetriever`, `VectorCypherRetriever`, `Text2CypherRetriever`, `HybridCypherRetriever`
- **KG construction pipeline**: `SimpleKGPipeline` extracts entities + relationships from documents automatically
- **Schema-aware Cypher generation**: `Text2CypherRetriever` reads graph schema and writes correct queries
- **LLM-agnostic**: swap between OpenAI, Anthropic, Vertex AI, Ollama, Cohere, Mistral via config
- **Embedder-agnostic**: same for embeddings

Result: Phases 1–4 of the original TS plan collapse to ~40% less code.

---

## 2. Tech stack

| Layer | Choice | Notes |
|-------|--------|-------|
| Language | Python 3.11+ | Required by `neo4j-graphrag` |
| Graph DB | Neo4j Desktop (local) | Community edition, includes GDS |
| GraphRAG framework | `neo4j-graphrag` (official) | https://github.com/neo4j/neo4j-graphrag-python |
| LLM provider | Configurable (start with Ollama → Claude/GPT-4o later) | Free local dev, swap for prod |
| Embeddings | Configurable (SentenceTransformers local → Vertex/OpenAI for quality) | Free local option |
| Data source | OpenAlex API | Free, no API key, ~240M works |
| HTTP client | `httpx` | Async-friendly |
| CLI | `typer` | Modern, type-hinted CLI |
| Config | `pydantic-settings` + `.env` | Same pattern as TS version |

---

## 3. Project structure

```
GraphRag-Python/
├── pyproject.toml                   # Dependencies + metadata
├── .env.example                     # Config template
├── .gitignore
├── README.md
├── docs/
│   ├── implementation-plan.md       # This document
│   └── queries.md                   # Sample Cypher queries
├── src/
│   └── research_graphrag/
│       ├── __init__.py
│       ├── config.py                # Settings (Neo4j URI, LLM choice, etc.)
│       ├── clients/
│       │   ├── __init__.py
│       │   ├── openalex.py          # OpenAlex API client
│       │   └── neo4j_client.py      # Driver singleton + helpers
│       ├── schema/
│       │   ├── __init__.py
│       │   ├── models.py            # Pydantic models for Paper, Author, Concept
│       │   └── cypher.py            # Schema DDL (constraints, indexes)
│       ├── ingest/
│       │   ├── __init__.py
│       │   ├── seed.py              # Topic-based BFS ingestion
│       │   ├── embed.py             # Batch abstract embedding
│       │   └── dedup.py             # MERGE-based deduplication
│       ├── algorithms/
│       │   ├── __init__.py
│       │   ├── pagerank.py          # Pure Cypher or GDS PageRank
│       │   ├── communities.py       # Louvain / modularity
│       │   ├── paths.py             # Shortest path, lineage
│       │   └── centrality.py        # Betweenness, degree
│       ├── retrieval/
│       │   ├── __init__.py
│       │   ├── retrievers.py        # Configured VectorCypherRetriever etc.
│       │   └── text2cypher.py       # NL → Cypher wrapper
│       ├── pipelines/
│       │   ├── __init__.py
│       │   ├── reading_list.py      # Curated reading order flow
│       │   ├── lineage.py           # Idea evolution flow
│       │   ├── gaps.py              # Research gap finder
│       │   └── research_rag.py      # General Q&A
│       └── cli.py                   # Typer CLI entry point
└── tests/
    ├── __init__.py
    ├── test_openalex.py
    ├── test_ingest.py
    └── test_retrievers.py
```

---

## 4. Dependencies (`pyproject.toml`)

```toml
[project]
name = "research-graphrag"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "neo4j-graphrag[openai,ollama,sentence-transformers]>=1.0.0",
    "neo4j>=5.27.0",
    "httpx>=0.27.0",
    "pydantic>=2.9.0",
    "pydantic-settings>=2.5.0",
    "typer>=0.12.0",
    "rich>=13.0.0",
    "tenacity>=8.2.0",           # Retry with backoff for API calls
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
    "ruff>=0.6.0",
    "mypy>=1.11.0",
]

[project.scripts]
research = "research_graphrag.cli:app"
```

Install:
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

---

## 5. Graph schema

### Nodes
```cypher
(:Paper {
  id: STRING,              // OpenAlex Work ID, e.g. "W2741809807"
  doi: STRING,
  title: STRING,
  abstract: STRING,
  year: INTEGER,
  citation_count: INTEGER,
  venue_id: STRING,
  openalex_type: STRING,   // article, book-chapter, etc.
  embedding: LIST<FLOAT>   // abstract embedding (for vector index)
})

(:Author {
  id: STRING,              // OpenAlex Author ID
  name: STRING,
  orcid: STRING,
  h_index: INTEGER,
  works_count: INTEGER
})

(:Institution {
  id: STRING,
  name: STRING,
  country: STRING,
  type: STRING             // education, company, government, etc.
})

(:Venue {
  id: STRING,
  name: STRING,
  type: STRING,            // journal, conference, repository
  issn: STRING
})

(:Concept {
  id: STRING,
  name: STRING,
  level: INTEGER,          // 0 (root) through 5 (specific)
  wikidata_url: STRING
})
```

### Relationships
```cypher
(:Paper)-[:CITES {year: INTEGER}]->(:Paper)
(:Paper)-[:AUTHORED_BY {position: INTEGER}]->(:Author)
(:Author)-[:AFFILIATED_WITH {year: INTEGER}]->(:Institution)
(:Paper)-[:ABOUT {score: FLOAT}]->(:Concept)
(:Paper)-[:PUBLISHED_IN]->(:Venue)
(:Concept)-[:PARENT_OF]->(:Concept)
```

### Constraints & indexes (Cypher, applied on first run)
```cypher
CREATE CONSTRAINT paper_id IF NOT EXISTS FOR (p:Paper) REQUIRE p.id IS UNIQUE;
CREATE CONSTRAINT author_id IF NOT EXISTS FOR (a:Author) REQUIRE a.id IS UNIQUE;
CREATE CONSTRAINT institution_id IF NOT EXISTS FOR (i:Institution) REQUIRE i.id IS UNIQUE;
CREATE CONSTRAINT venue_id IF NOT EXISTS FOR (v:Venue) REQUIRE v.id IS UNIQUE;
CREATE CONSTRAINT concept_id IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE;

CREATE INDEX paper_year IF NOT EXISTS FOR (p:Paper) ON (p.year);
CREATE INDEX paper_title IF NOT EXISTS FOR (p:Paper) ON (p.title);
CREATE FULLTEXT INDEX paper_text IF NOT EXISTS FOR (p:Paper) ON EACH [p.title, p.abstract];

CALL db.index.vector.createNodeIndex(
  'paper_abstracts',
  'Paper',
  'embedding',
  768,                    // SentenceTransformers default; change if swap embedder
  'cosine'
);
```

---

## 6. Environment (`.env.example`)

```env
# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=changeme
NEO4J_DATABASE=neo4j

# LLM provider selection
LLM_PROVIDER=ollama                  # ollama | openai | anthropic
LLM_MODEL=llama3.1:8b

# Embeddings
EMBEDDING_PROVIDER=sentence-transformers  # sentence-transformers | openai
EMBEDDING_MODEL=all-MiniLM-L6-v2     # 384-dim, fast; or all-mpnet-base-v2 (768-dim, better)

# Optional paid providers
OPENAI_API_KEY=
ANTHROPIC_API_KEY=

# OpenAlex
OPENALEX_USER_AGENT_EMAIL=your-email@example.com   # Required for polite pool

# Ingestion limits
MAX_PAPERS_PER_SEED=500
MAX_CITATION_DEPTH=1
BATCH_EMBEDDING_SIZE=32
```

---

## 7. Phased implementation

### Phase 0 — Scaffold (½ day)
**Deliverables:** working venv, deps installed, Neo4j Desktop connected, schema DDL runs.

- [ ] `pyproject.toml` with deps locked
- [ ] `config.py` using `pydantic-settings` to load `.env`
- [ ] `clients/neo4j_client.py` — singleton driver, context manager for sessions
- [ ] `schema/cypher.py` — idempotent DDL apply function
- [ ] `cli.py` with `research init` command (runs DDL)
- [ ] Smoke test: `research init` creates all constraints + vector index

### Phase 1 — Ingestion MVP (2 days)
**Deliverables:** seed a topic, see citation graph in Neo4j Browser.

- [ ] `clients/openalex.py` — async client with:
  - `search_works(query: str, per_page: int) -> AsyncIterator[dict]`
  - `fetch_work(openalex_id: str) -> dict`
  - `fetch_author(openalex_id: str) -> dict`
  - Polite pool header, 10 req/sec rate limit, retry with `tenacity`
- [ ] `schema/models.py` — Pydantic models: `Paper`, `Author`, `Institution`, `Venue`, `Concept`
- [ ] `ingest/seed.py`:
  - `seed_topic(topic: str, max_papers: int, depth: int)` — BFS expansion
  - Uses `MERGE` everywhere (idempotent, dedup by OpenAlex ID)
- [ ] `ingest/embed.py`:
  - Batch embed abstracts using configured embedder
  - `neo4j-graphrag` provides `SentenceTransformerEmbeddings` out of the box
- [ ] CLI command: `research seed "diffusion models" --max-papers 200`
- [ ] Verify in Neo4j Browser:
  ```cypher
  MATCH (p:Paper)-[:CITES]->(q:Paper) RETURN p, q LIMIT 50
  ```

### Phase 2 — Retrieval (1-2 days)
**Deliverables:** four working retrievers callable from CLI.

Leverage `neo4j-graphrag` retrievers directly:

```python
from neo4j_graphrag.retrievers import (
    VectorRetriever,
    VectorCypherRetriever,
    HybridRetriever,
    Text2CypherRetriever,
)

# 1. Pure vector search over abstracts
vector_retriever = VectorRetriever(
    driver=driver,
    index_name="paper_abstracts",
    embedder=embedder,
)

# 2. Vector + graph expansion (THE key retriever for GraphRAG)
vector_cypher_retriever = VectorCypherRetriever(
    driver=driver,
    index_name="paper_abstracts",
    embedder=embedder,
    retrieval_query="""
        MATCH (p:Paper)-[:ABOUT]->(c:Concept)
        OPTIONAL MATCH (p)-[:AUTHORED_BY]->(a:Author)
        OPTIONAL MATCH (p)-[:CITES]->(cited:Paper)
        RETURN p.title AS title,
               p.abstract AS abstract,
               p.year AS year,
               collect(DISTINCT c.name) AS concepts,
               collect(DISTINCT a.name) AS authors,
               collect(DISTINCT cited.title)[..5] AS cites
    """,
)

# 3. Hybrid: vector + full-text over title/abstract
hybrid_retriever = HybridRetriever(
    driver=driver,
    vector_index_name="paper_abstracts",
    fulltext_index_name="paper_text",
    embedder=embedder,
)

# 4. Natural language → Cypher
text2cypher_retriever = Text2CypherRetriever(
    driver=driver,
    llm=llm,
    neo4j_schema=get_schema_description(),
)
```

- [ ] `retrieval/retrievers.py` — factory functions for each retriever
- [ ] CLI: `research retrieve --mode vector|hybrid|graph|cypher "question"`
- [ ] Print source papers with OpenAlex links in the output

### Phase 3 — Research-specific pipelines (2 days)
**Deliverables:** four purpose-built flows.

All flows use `neo4j_graphrag.generation.GraphRAG`:

```python
from neo4j_graphrag.generation import GraphRAG

rag = GraphRAG(retriever=vector_cypher_retriever, llm=llm)
response = rag.search(
    query_text=question,
    return_context=True,
)
```

- [ ] `pipelines/reading_list.py` — `reading_list(topic, level)`:
  1. Find concept subgraph
  2. Run Cypher PageRank approximation to find seminal papers
  3. Sort by year
  4. LLM synthesizes ordered recommendation with rationale
- [ ] `pipelines/lineage.py` — `trace_lineage(from_paper, to_paper)`:
  1. Find shortest path in CITES graph
  2. Fetch path papers
  3. LLM narrates the evolution
- [ ] `pipelines/gaps.py` — `find_gaps(concept_a, concept_b)`:
  1. Papers tagged with both concepts → underexplored intersection
  2. LLM interprets findings
- [ ] `pipelines/research_rag.py` — general Q&A using `VectorCypherRetriever`
- [ ] CLI commands:
  - `research reading-list "diffusion models" --level beginner`
  - `research lineage --from W123 --to W456`
  - `research gaps "NLP" "robotics"`
  - `research ask "..."`

### Phase 4 — Graph algorithms (1 day, optional)
**Deliverables:** analytics commands using GDS plugin.

Requires Neo4j Desktop with GDS plugin installed.

- [ ] `algorithms/pagerank.py`:
  ```python
  # Project the citation subgraph then run PageRank
  CALL gds.graph.project(
    'citations',
    'Paper',
    'CITES'
  )
  CALL gds.pageRank.stream('citations')
  YIELD nodeId, score
  RETURN gds.util.asNode(nodeId).title AS title, score
  ORDER BY score DESC LIMIT 20
  ```
- [ ] `algorithms/communities.py` — Louvain for research clusters
- [ ] `algorithms/paths.py` — shortest path, allShortestPaths
- [ ] `algorithms/centrality.py` — betweenness for bridge papers
- [ ] CLI: `research analyze --algo pagerank|louvain|betweenness --concept "diffusion models"`

### Phase 5 — Polish & docs (1 day)
- [ ] README with examples
- [ ] `docs/queries.md` with curated Cypher patterns
- [ ] Basic tests for clients/ingest
- [ ] Rate limit handling for abstract embedding in large batches
- [ ] Export/import: dump + restore graph state

---

## 8. Key design decisions

### LLM/embedder choice strategy

Start **local & free** to validate the pipeline, swap for production quality later:

| Phase | LLM | Embedder | Why |
|-------|-----|----------|-----|
| Dev (Phase 0-3) | Ollama `llama3.1:8b` | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) | Free, no API limits, ~1 sec per call |
| Polish (Phase 5) | Ollama `llama3.1:70b` OR Claude Haiku | `all-mpnet-base-v2` (768-dim) | Better quality |
| Production | Claude Sonnet OR GPT-4o | OpenAI `text-embedding-3-large` (3072-dim) | Best quality |

**Critical:** swapping embedders changes vector dimension → must drop & recreate `paper_abstracts` index.

### Dedup strategy
All writes use `MERGE` on OpenAlex ID as primary key. The same paper arriving via different seed topics updates properties idempotently.

### Graph size control
- Cap CITES expansion per node (`LIMIT 100` most-cited references)
- Hard ceiling: `MAX_PAPERS_PER_SEED` (default 500)
- Check current node count before ingest; warn if approaching limits

### Rate limiting
- OpenAlex: 10 req/sec in polite pool (requires email in User-Agent)
- Exponential backoff via `tenacity` on 429/5xx
- Abstract embedding: batch of 32 per API call

### Error handling
- Network errors: retry 3x with backoff
- Missing fields from OpenAlex: skip the field, log warning, continue
- Dead OpenAlex IDs: mark paper as partial, don't crash ingest

---

## 9. Concrete sprint plan

### Sprint 1 (3-4 days): Working demo end-to-end
Phase 0 + Phase 1 + minimal Phase 2 (just `VectorCypherRetriever`) + basic CLI.

Demo: `research seed "diffusion models" && research ask "Give me a reading list for diffusion models"`

### Sprint 2 (2-3 days): Specialized retrievers & flows
Full Phase 2 + Phase 3 (all four pipelines).

### Sprint 3 (1-2 days): Analytics
Phase 4 (GDS algorithms).

### Sprint 4 (1 day): Polish
Phase 5 (docs, tests, README).

**Total: ~7-10 working days for a polished, production-shaped system.**

---

## 10. Migration from TypeScript version

The TS project stays as-is (demos the Genkit approach, already in GitHub). This Python project is a **parallel rewrite**, not a port — different architecture.

**What carries over:**
- Conceptual approach (vector + graph traversal)
- OpenAlex as data source (new, wasn't in TS version)
- Neo4j as store (but connection config moves to local Desktop)

**What doesn't:**
- Code (different language, different framework)
- Genkit dev UI (replaced by CLI + `neo4j-graphrag` built-in tracing)
- Entity extraction approach (replaced by OpenAlex structured data + optional `SimpleKGPipeline` for abstract-derived concepts)

---

## 11. Success criteria

A working system that can answer these questions with graph-grounded citations:

1. *"Which 5 papers should I read to understand diffusion models?"* → reading list with justification
2. *"Trace the evolution of attention mechanisms from 2015 to 2020"* → citation path with narrative
3. *"What are the most influential papers in multi-agent reinforcement learning?"* → PageRank-ranked list
4. *"Which research communities exist within NLP?"* → Louvain-detected clusters
5. *"Find underexplored gaps between computer vision and causal inference"* → intersection analysis
6. *"Who are the bridge authors between theoretical and applied ML?"* → betweenness ranking

Each answer must include linked OpenAlex IDs so the user can verify.

---

## 12. Open questions (resolve before Sprint 1)

1. **Neo4j Desktop installed locally?** → prereq for all of this
2. **Ollama installed?** → for free local LLM during dev (otherwise switch default to OpenAI/Anthropic + budget)
3. **Seed topic for first demo?** → pick one to focus on ("diffusion models" is a strong default — dense citation network, recent enough to be interesting)
4. **UI?** → CLI-only for MVP. If graph visualization wanted later, pick between Neo4j Bloom (built-in), custom web UI (React + vis.js), or Jupyter notebooks with `pyvis`.

---

## Next step

Install Neo4j Desktop (if not already), then run:

```bash
cd GraphRag-Python
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -e ".[dev]"
cp .env.example .env
# edit .env with local Neo4j password
research init                  # apply schema
research seed "diffusion models" --max-papers 100
research ask "What are the foundational diffusion model papers?"
```
