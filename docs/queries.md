# Curated Cypher queries

Snippets for Neo4j Browser (`:use neo4j` first). All queries assume the schema defined in [`src/research_graphrag/schema/cypher.py`](../src/research_graphrag/schema/cypher.py).

## Graph overview

```cypher
MATCH (n)
RETURN labels(n)[0] AS label, count(*) AS count
ORDER BY count DESC;
```

```cypher
MATCH ()-[r]->()
RETURN type(r) AS rel_type, count(*) AS count
ORDER BY count DESC;
```

```cypher
MATCH (p:Paper)
RETURN
  count(p) AS total_papers,
  count(p.title) AS with_title,
  count(p.abstract) AS with_abstract,
  count(p.embedding) AS with_embedding;
```

## Finding papers

### Top-cited papers (by OpenAlex global citation count)

```cypher
MATCH (p:Paper)
WHERE p.title IS NOT NULL
RETURN p.id, p.title, p.year, p.citation_count
ORDER BY p.citation_count DESC
LIMIT 20;
```

### Top-cited *within this corpus* (in-degree on :CITES)

```cypher
MATCH (p:Paper)
WHERE p.title IS NOT NULL
OPTIONAL MATCH (p)<-[:CITES]-(citer:Paper)
WITH p, count(citer) AS local_citations
ORDER BY local_citations DESC
LIMIT 20
RETURN p.id, p.title, p.year, local_citations;
```

### Papers on a specific concept

```cypher
MATCH (c:Concept)
WHERE toLower(c.name) CONTAINS 'diffusion'
MATCH (p:Paper)-[:ABOUT]->(c)
WHERE p.title IS NOT NULL
RETURN DISTINCT p.id, p.title, p.year, p.citation_count
ORDER BY p.citation_count DESC
LIMIT 25;
```

## Exploring a single paper

### A paper and everything it cites

```cypher
MATCH (p:Paper {id: 'W4312933868'})-[:CITES]->(cited:Paper)
RETURN p, cited
LIMIT 50;
```

### A paper's concepts, authors, venue

```cypher
MATCH (p:Paper {id: 'W4312933868'})
OPTIONAL MATCH (p)-[:ABOUT]->(c:Concept)
OPTIONAL MATCH (p)-[:AUTHORED_BY]->(a:Author)
OPTIONAL MATCH (p)-[:PUBLISHED_IN]->(v:Venue)
RETURN p.title AS title,
       collect(DISTINCT c.name) AS concepts,
       collect(DISTINCT a.name) AS authors,
       collect(DISTINCT v.name) AS venue;
```

### Neighborhood (2 hops of citation)

```cypher
MATCH path = (p:Paper {id: 'W4312933868'})-[:CITES*1..2]-(other:Paper)
WHERE other.title IS NOT NULL
RETURN path
LIMIT 100;
```

## Authors & institutions

### Most prolific authors in this corpus

```cypher
MATCH (a:Author)<-[:AUTHORED_BY]-(p:Paper)
WHERE p.title IS NOT NULL
WITH a, count(DISTINCT p) AS paper_count
ORDER BY paper_count DESC
LIMIT 15
RETURN a.name, paper_count;
```

### Institutions by paper volume

```cypher
MATCH (i:Institution)<-[:AFFILIATED_WITH]-(a:Author)<-[:AUTHORED_BY]-(p:Paper)
WHERE p.title IS NOT NULL
WITH i, count(DISTINCT p) AS paper_count
ORDER BY paper_count DESC
LIMIT 15
RETURN i.name, i.country, paper_count;
```

## Concept analysis

### Concept co-occurrence (papers sharing both)

```cypher
MATCH (ca:Concept {name: 'Generative model'})<-[:ABOUT]-(p:Paper)-[:ABOUT]->(cb:Concept)
WHERE cb <> ca
WITH cb, count(DISTINCT p) AS overlap
ORDER BY overlap DESC
LIMIT 15
RETURN cb.name, overlap;
```

### Year distribution on a concept

```cypher
MATCH (c:Concept)
WHERE toLower(c.name) CONTAINS 'diffusion'
MATCH (p:Paper)-[:ABOUT]->(c)
WHERE p.year IS NOT NULL
RETURN p.year, count(*) AS papers
ORDER BY p.year;
```

## Stubs vs fully-ingested papers

Papers created as stubs (only `id`, because they were referenced but not fetched):

```cypher
MATCH (p:Paper)
WHERE p.title IS NULL
RETURN count(p) AS stub_count;
```

Fully-ingested papers with their abstract status:

```cypher
MATCH (p:Paper)
WHERE p.title IS NOT NULL
RETURN
  count(p) AS fully_ingested,
  count(p.abstract) AS with_abstract,
  count(p.embedding) AS with_embedding;
```

## Vector similarity (without Python)

### Find papers similar to a given paper's abstract

```cypher
MATCH (seed:Paper {id: 'W4312933868'})
CALL db.index.vector.queryNodes('paper_abstracts', 10, seed.embedding)
YIELD node, score
WHERE node.id <> seed.id
RETURN node.id, node.title, node.year, score
ORDER BY score DESC;
```

## GDS algorithms (requires plugin)

### Create the in-memory projection

```cypher
CALL gds.graph.drop('citations', false);

CALL gds.graph.project(
  'citations',
  'Paper',
  {CITES: {orientation: 'UNDIRECTED'}}
);
```

### PageRank top 20

```cypher
CALL gds.pageRank.stream('citations')
YIELD nodeId, score
WITH gds.util.asNode(nodeId) AS p, score
WHERE p.title IS NOT NULL
RETURN p.id, p.title, p.year, score
ORDER BY score DESC
LIMIT 20;
```

### Louvain communities (sized)

```cypher
CALL gds.louvain.stream('citations')
YIELD nodeId, communityId
WITH gds.util.asNode(nodeId) AS p, communityId
WHERE p.title IS NOT NULL
WITH communityId, collect(p) AS papers
WHERE size(papers) >= 5
RETURN communityId, size(papers) AS size,
       [x IN papers | x.title][..5] AS sample
ORDER BY size DESC
LIMIT 10;
```

### Bridge papers (betweenness, concept-filtered)

```cypher
CALL gds.betweenness.stream('citations')
YIELD nodeId, score
WITH gds.util.asNode(nodeId) AS p, score
WHERE p.title IS NOT NULL
  AND EXISTS { (p)-[:ABOUT]->(c:Concept) WHERE toLower(c.name) CONTAINS 'generative' }
RETURN p.id, p.title, p.year, score
ORDER BY score DESC
LIMIT 15;
```

## Housekeeping

### Drop vector index (needed if you change embedding dimensions)

```cypher
DROP INDEX paper_abstracts;
```

### Nuke everything (danger)

```cypher
MATCH (n) DETACH DELETE n;
```
