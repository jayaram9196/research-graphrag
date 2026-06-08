from research_graphrag.clients.openalex import (
    MAX_REFERENCES_PER_PAPER,
    _normalize_author,
    _normalize_work,
    _reconstruct_abstract,
    _strip_id,
)


def test_strip_id_removes_openalex_prefix():
    assert _strip_id("https://openalex.org/W2741809807") == "W2741809807"


def test_strip_id_removes_orcid_prefix():
    assert _strip_id("https://orcid.org/0000-0001-2345-6789") == "0000-0001-2345-6789"


def test_strip_id_handles_none():
    assert _strip_id(None) is None


def test_strip_id_passes_through_already_short_id():
    assert _strip_id("W123") == "W123"


def test_reconstruct_abstract_none_or_empty():
    assert _reconstruct_abstract(None) is None
    assert _reconstruct_abstract({}) is None


def test_reconstruct_abstract_orders_by_position():
    inverted = {"the": [0, 3], "cat": [1], "sat": [2]}
    assert _reconstruct_abstract(inverted) == "the cat sat the"


def test_normalize_work_basic_fields():
    work = {
        "id": "https://openalex.org/W1",
        "doi": "https://doi.org/10.1/x",
        "title": "Test Paper",
        "publication_year": 2020,
        "cited_by_count": 42,
        "type": "article",
        "abstract_inverted_index": {"hello": [0], "world": [1]},
    }
    result = _normalize_work(work)
    assert result["id"] == "W1"
    assert result["doi"] == "https://doi.org/10.1/x"
    assert result["title"] == "Test Paper"
    assert result["year"] == 2020
    assert result["citation_count"] == 42
    assert result["openalex_type"] == "article"
    assert result["abstract"] == "hello world"


def test_normalize_work_extracts_authors_and_institutions():
    work = {
        "id": "https://openalex.org/W2",
        "title": "X",
        "authorships": [
            {
                "author_position": "first",
                "author": {
                    "id": "https://openalex.org/A1",
                    "display_name": "Alice",
                    "orcid": "https://orcid.org/0000-0001-0000-0000",
                },
                "institutions": [
                    {
                        "id": "https://openalex.org/I1",
                        "display_name": "Uni of X",
                        "country_code": "US",
                        "type": "education",
                    }
                ],
            }
        ],
    }
    author = _normalize_work(work)["authors"][0]
    assert author["id"] == "A1"
    assert author["name"] == "Alice"
    assert author["orcid"] == "0000-0001-0000-0000"
    assert author["position"] == "first"
    assert author["institutions"][0]["id"] == "I1"
    assert author["institutions"][0]["country"] == "US"


def test_normalize_work_extracts_concepts():
    work = {
        "id": "https://openalex.org/W3",
        "title": "X",
        "concepts": [
            {
                "id": "https://openalex.org/C1",
                "display_name": "Machine learning",
                "level": 1,
                "score": 0.91,
                "wikidata": "https://www.wikidata.org/wiki/Q2539",
            }
        ],
    }
    concept = _normalize_work(work)["concepts"][0]
    assert concept["id"] == "C1"
    assert concept["name"] == "Machine learning"
    assert concept["level"] == 1
    assert concept["score"] == 0.91


def test_normalize_work_strips_reference_prefixes():
    work = {
        "id": "https://openalex.org/W4",
        "title": "X",
        "referenced_works": [
            "https://openalex.org/W10",
            "https://openalex.org/W20",
        ],
    }
    assert _normalize_work(work)["referenced_works"] == ["W10", "W20"]


def test_normalize_work_caps_references_at_max():
    too_many = [f"https://openalex.org/W{i}" for i in range(MAX_REFERENCES_PER_PAPER + 50)]
    work = {"id": "https://openalex.org/W5", "title": "X", "referenced_works": too_many}
    assert len(_normalize_work(work)["referenced_works"]) == MAX_REFERENCES_PER_PAPER


def test_normalize_work_handles_missing_abstract():
    work = {"id": "https://openalex.org/W6", "title": "X"}
    assert _normalize_work(work)["abstract"] is None


def test_normalize_work_extracts_venue_when_present():
    work = {
        "id": "https://openalex.org/W7",
        "title": "X",
        "primary_location": {
            "source": {
                "id": "https://openalex.org/S1",
                "display_name": "NeurIPS",
                "type": "conference",
                "issn_l": "1234-5678",
            }
        },
    }
    venue = _normalize_work(work)["venue"]
    assert venue is not None
    assert venue["id"] == "S1"
    assert venue["name"] == "NeurIPS"
    assert venue["type"] == "conference"


def test_normalize_work_returns_no_venue_when_missing():
    work = {"id": "https://openalex.org/W8", "title": "X"}
    assert _normalize_work(work)["venue"] is None


def test_normalize_author_pulls_h_index_from_summary_stats():
    author = {
        "id": "https://openalex.org/A42",
        "display_name": "Bob",
        "orcid": "https://orcid.org/0000-0001-2222-3333",
        "works_count": 10,
        "summary_stats": {"h_index": 5},
    }
    result = _normalize_author(author)
    assert result["id"] == "A42"
    assert result["name"] == "Bob"
    assert result["orcid"] == "0000-0001-2222-3333"
    assert result["h_index"] == 5
    assert result["works_count"] == 10
