from pydantic import BaseModel


class Concept(BaseModel):
    id: str
    name: str | None = None
    level: int | None = None
    score: float | None = None
    wikidata_url: str | None = None


class Institution(BaseModel):
    id: str
    name: str | None = None
    country: str | None = None
    type: str | None = None


class AuthorAffiliation(BaseModel):
    id: str
    name: str | None = None
    orcid: str | None = None
    position: str | None = None
    institutions: list[Institution] = []


class Author(BaseModel):
    id: str
    name: str | None = None
    orcid: str | None = None
    h_index: int | None = None
    works_count: int | None = None


class Venue(BaseModel):
    id: str
    name: str | None = None
    type: str | None = None
    issn: str | None = None


class Paper(BaseModel):
    id: str
    doi: str | None = None
    title: str | None = None
    abstract: str | None = None
    year: int | None = None
    citation_count: int | None = 0
    openalex_type: str | None = None
    venue: Venue | None = None
    authors: list[AuthorAffiliation] = []
    concepts: list[Concept] = []
    referenced_works: list[str] = []
