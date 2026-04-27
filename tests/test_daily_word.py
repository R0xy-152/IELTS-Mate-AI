"""Tests for GET /api/daily_word."""
import json

import main


def _seed_word(db, word, tags, **kwargs):
    w = main.DBWord(
        word=word,
        pos=kwargs.get("pos", "n."),
        cn=kwargs.get("cn", f"中文-{word}"),
        en_definition=kwargs.get("en_definition", f"def of {word}"),
        context=json.dumps(tags),
        example=kwargs.get("example", f"Use {word} in a sentence."),
    )
    db.add(w)
    db.commit()
    db.refresh(w)
    return w


def test_daily_word_uses_db_when_word_matches_no_tags(client, db_session):
    """No saved preferences, but the DB has a word — it should be picked."""
    _seed_word(db_session, "alpha", [])
    r = client.get("/api/daily_word")
    assert r.status_code == 200
    body = r.json()
    assert body["data"]["word"] == "alpha"


def test_daily_word_filters_by_selected_tags(client, db_session):
    """User picked Music — ballad (Music) wins over thermal (Technology)."""
    _seed_word(db_session, "thermal", ["Technology"])
    _seed_word(db_session, "ballad", ["Music"])
    db_session.add(main.UserPreference(selected_tags=json.dumps(["Music"])))
    db_session.commit()

    r = client.get("/api/daily_word")
    assert r.status_code == 200
    assert r.json()["data"]["word"] == "ballad"


def test_daily_word_caches_within_same_day(client, db_session):
    """Two consecutive calls on the same day return the same word."""
    _seed_word(db_session, "alpha", [])
    _seed_word(db_session, "beta", [])
    _seed_word(db_session, "gamma", [])

    first = client.get("/api/daily_word").json()
    second = client.get("/api/daily_word").json()
    third = client.get("/api/daily_word").json()

    assert first["data"]["word"] == second["data"]["word"] == third["data"]["word"]
    assert second["source"] == "cached"
    assert third["source"] == "cached"


def test_daily_word_safety_net_falls_back_to_random(
    client, db_session, monkeypatch
):
    """User has a tag that matches nothing AND AI suggestion fails -> random fallback."""
    _seed_word(db_session, "orphan", ["Society"])
    db_session.add(main.UserPreference(
        selected_tags=json.dumps(["NonexistentTag"])
    ))
    db_session.commit()

    def _boom(tags):
        raise RuntimeError("AI down")
    monkeypatch.setattr(main, "ai_suggest_word_for_tags", _boom)

    r = client.get("/api/daily_word")
    assert r.status_code == 200
    assert r.json()["data"]["word"] == "orphan"


def test_daily_word_503_when_db_empty_and_ai_unavailable(client, monkeypatch):
    """Empty DB, no API key configured, AI suggestion fails -> 503."""
    monkeypatch.setattr(main, "GEMINI_API_KEY", None)

    def _boom(tags):
        raise RuntimeError("AI down")
    monkeypatch.setattr(main, "ai_suggest_word_for_tags", _boom)

    r = client.get("/api/daily_word")
    assert r.status_code == 503


def test_daily_word_uses_ai_suggestion_when_db_lacks_tag_match(
    client, db_session, monkeypatch
):
    """User has tag preference, no matching word in DB -> AI suggests one and we generate it."""
    db_session.add(main.UserPreference(selected_tags=json.dumps(["Programming"])))
    db_session.commit()

    monkeypatch.setattr(main, "ai_suggest_word_for_tags", lambda tags: "algorithm")
    monkeypatch.setattr(main, "fetch_dictionary", lambda w: {
        "meanings": [{
            "partOfSpeech": "noun",
            "definitions": [{"definition": "a procedure", "example": "ex"}],
        }],
    })
    monkeypatch.setattr(main, "ai_translate", lambda w, d: "算法")
    monkeypatch.setattr(
        main, "ai_extract_topic_tags",
        lambda w, d, allowed: ["Programming"],
    )

    r = client.get("/api/daily_word")
    assert r.status_code == 200
    assert r.json()["data"]["word"] == "algorithm"

    stored = db_session.query(main.DBWord).filter(
        main.DBWord.word == "algorithm"
    ).first()
    assert stored is not None
