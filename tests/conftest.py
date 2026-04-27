"""Shared pytest fixtures.

Goals:
  - Each test gets an isolated in-memory SQLite database.
  - All external dependencies (DictionaryAPI HTTP + Gemini AI calls) are
    auto-stubbed with safe defaults. Tests can override individual stubs
    via monkeypatch when they need to test specific behavior.
  - No API key, no network, no real Gemini configuration is required.
"""
import pathlib
import sys

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import main  # noqa: E402  (path-dependent import)


@pytest.fixture
def db_engine():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    main.Base.metadata.create_all(bind=engine)
    yield engine
    main.Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture
def db_session(db_engine):
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)
    session = TestingSession()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def client(db_engine):
    """FastAPI TestClient with the in-memory DB injected via dependency override."""
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)

    def override_get_db():
        db = TestingSession()
        try:
            yield db
        finally:
            db.close()

    main.app.dependency_overrides[main.get_db] = override_get_db
    with TestClient(main.app) as c:
        yield c
    main.app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def stub_externals(monkeypatch):
    """Replace every external dependency with deterministic stubs.

    Default behavior:
      - fetch_dictionary returns None  -> forces the AI fallback path
      - all ai_* helpers return safe deterministic values

    Tests can override any of these via monkeypatch.setattr(main, "...", ...).
    """
    monkeypatch.setattr(main, "fetch_dictionary", lambda w: None)
    monkeypatch.setattr(main, "ai_translate", lambda w, d: f"中文-{w}")
    monkeypatch.setattr(main, "ai_generate_example", lambda w: f"Example using {w}.")
    monkeypatch.setattr(
        main,
        "ai_base_info",
        lambda w: {"pos": "noun", "cn": f"中文-{w}", "en_definition": f"def of {w}"},
    )
    monkeypatch.setattr(
        main, "ai_extract_topic_tags", lambda w, d, allowed: []
    )
    monkeypatch.setattr(
        main, "ai_suggest_word_for_tags", lambda tags: "sustainability"
    )
    # By default, treat the API key as configured so the "last resort"
    # generation path in /api/daily_word is reachable. Tests that want to
    # simulate the missing-key case can override this.
    monkeypatch.setattr(main, "GEMINI_API_KEY", "test-key")
