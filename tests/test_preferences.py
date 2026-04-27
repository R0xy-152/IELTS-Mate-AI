"""Tests for /api/preferences and /api/pre-cache-tags."""
import json

import main


def test_get_preferences_empty_returns_empty_list(client):
    r = client.get("/api/preferences")
    assert r.status_code == 200
    assert r.json() == {"selected_tags": []}


def test_save_then_get_preferences_round_trip(client):
    r = client.post("/api/preferences", json={"selected_tags": ["Music", "Sports"]})
    assert r.status_code == 200
    assert sorted(r.json()["selected_tags"]) == ["Music", "Sports"]

    r = client.get("/api/preferences")
    assert sorted(r.json()["selected_tags"]) == ["Music", "Sports"]


def test_save_preferences_overwrites_previous(client):
    client.post("/api/preferences", json={"selected_tags": ["Music"]})
    client.post("/api/preferences", json={"selected_tags": ["Sports", "Travel"]})

    r = client.get("/api/preferences")
    assert sorted(r.json()["selected_tags"]) == ["Sports", "Travel"]


def test_save_preferences_clears_daily_word_cache(client, db_session):
    """Updating tags should drop the cached daily-word so the next call recomputes."""
    db_session.add(main.UserPreference(
        selected_tags=json.dumps(["Music"]),
        last_daily_word_date="2099-01-01",
        last_daily_word_id=42,
    ))
    db_session.commit()

    r = client.post("/api/preferences", json={"selected_tags": ["Sports"]})
    assert r.status_code == 200

    db_session.expire_all()
    pref = db_session.query(main.UserPreference).first()
    assert pref.last_daily_word_date is None
    assert pref.last_daily_word_id is None


def test_pre_cache_tags_503_when_no_api_key(client, monkeypatch):
    monkeypatch.setattr(main, "GEMINI_API_KEY", None)
    r = client.get("/api/pre-cache-tags")
    assert r.status_code == 503


def test_root_serves_index_html(client):
    """GET / should serve the SPA, NOT expose project files."""
    r = client.get("/")
    assert r.status_code == 200
    assert "<!DOCTYPE html>" in r.text or "<html" in r.text


def test_static_mount_does_not_expose_dotenv(client):
    """Critical: /static/ MUST NOT serve files outside the static folder."""
    r = client.get("/.env")
    assert r.status_code == 404
    r = client.get("/main.py")
    assert r.status_code == 404
    r = client.get("/ielts_mate.db")
    assert r.status_code == 404
