from collections.abc import Iterator
from contextlib import contextmanager

from neo4j import Driver, GraphDatabase, Session

from ..config import get_settings

_driver: Driver | None = None


def get_driver() -> Driver:
    global _driver
    if _driver is None:
        s = get_settings()
        _driver = GraphDatabase.driver(
            s.neo4j_uri,
            auth=(s.neo4j_username, s.neo4j_password),
        )
        _driver.verify_connectivity()
    return _driver


@contextmanager
def get_session() -> Iterator[Session]:
    driver = get_driver()
    database = get_settings().neo4j_database
    session = driver.session(database=database)
    try:
        yield session
    finally:
        session.close()


def close_driver() -> None:
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None
