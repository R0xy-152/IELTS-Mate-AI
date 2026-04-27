"""Tests for GET /api/lookup/{word}."""
import json

import main


def test_lookup_cache_hit_skips_external_calls(client, db_session, monkeypatch):
    """Word already in DB should be returned without hitting any external API."""
    db_session.add(main.DBWord(
        word="cached",
        pos="n.",
        cn="缓存",
        en_definition="stored",
        context=json.dumps(["Technology"]),
        example="It is cached.",
    ))
    db_session.commit()

    def _boom(*a, **kw):
        raise AssertionError("external call should not be made on cache hit")

    monkeypatch.setattr(main, "fetch_dictionary", _boom)
    monkeypatch.setattr(main, "ai_base_info", _boom)

    r = client.get("/api/lookup/cached")
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "database"
    assert body["data"]["word"] == "cached"
    assert body["data"]["tags"] == ["Technology"]
    assert body["data"]["example"] == "It is cached."


def test_lookup_dict_api_success_path(client, monkeypatch):
    """When DictionaryAPI returns data, ai_base_info is NOT called."""
    monkeypatch.setattr(main, "fetch_dictionary", lambda w: {
        "meanings": [{
            "partOfSpeech": "noun",
            "definitions": [{
                "definition": "a small test thing",
                "example": "a test sentence",
            }],
        }],
    })
    monkeypatch.setattr(main, "ai_translate", lambda w, d: "测试")
    monkeypatch.setattr(main, "ai_extract_topic_tags",
                        lambda w, d, allowed: ["Technology"])

    def _should_not_run(*a, **kw):
        raise AssertionError("ai_base_info should not be used when DictAPI succeeds")
    monkeypatch.setattr(main, "ai_base_info", _should_not_run)

    r = client.get("/api/lookup/testword")
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "generated"
    assert body["data"]["pos"] == "noun"
    assert body["data"]["cn"] == "测试"
    assert body["data"]["en_definition"] == "a small test thing"
    assert body["data"]["example"] == "a test sentence"
    assert body["data"]["tags"] == ["Technology"]


def test_lookup_dict_api_fail_falls_back_to_ai(client, monkeypatch):
    """When DictionaryAPI fails, ai_base_info supplies pos/cn/en_definition."""
    monkeypatch.setattr(main, "fetch_dictionary", lambda w: None)
    monkeypatch.setattr(main, "ai_base_info", lambda w: {
        "pos": "noun",
        "cn": "降级",
        "en_definition": "fallback definition",
    })
    monkeypatch.setattr(main, "ai_extract_topic_tags",
                        lambda w, d, allowed: ["Society"])

    r = client.get("/api/lookup/anything")
    assert r.status_code == 200
    data = r.json()["data"]
    assert data["pos"] == "noun"
    assert data["cn"] == "降级"
    assert data["en_definition"] == "fallback definition"
    assert data["tags"] == ["Society"]


def test_lookup_full_failure_returns_502(client, monkeypatch):
    """Both DictionaryAPI and ai_base_info fail -> 502."""
    def _boom(*a, **kw):
        raise RuntimeError("AI down")

    monkeypatch.setattr(main, "fetch_dictionary", lambda w: None)
    monkeypatch.setattr(main, "ai_base_info", _boom)

    r = client.get("/api/lookup/willfail")
    assert r.status_code == 502


def test_lookup_persists_word_for_future_requests(client, db_session, monkeypatch):
    """A successfully generated word should be stored so the next call hits cache."""
    monkeypatch.setattr(main, "fetch_dictionary", lambda w: {
        "meanings": [{
            "partOfSpeech": "adj.",
            "definitions": [{"definition": "lasting", "example": ""}],
        }],
    })
    monkeypatch.setattr(main, "ai_translate", lambda w, d: "持久的")
    monkeypatch.setattr(main, "ai_extract_topic_tags",
                        lambda w, d, allowed: ["Environment"])

    first = client.get("/api/lookup/sustainable").json()
    assert first["source"] == "generated"

    stored = db_session.query(main.DBWord).filter(
        main.DBWord.word == "sustainable"
    ).first()
    assert stored is not None
    assert json.loads(stored.context) == ["Environment"]

    # Second call should be a cache hit; if any AI fired we'd see the assert.
    def _boom(*a, **kw):
        raise AssertionError("should be a cache hit")
    monkeypatch.setattr(main, "fetch_dictionary", _boom)
    monkeypatch.setattr(main, "ai_translate", _boom)
    monkeypatch.setattr(main, "ai_extract_topic_tags", _boom)

    second = client.get("/api/lookup/sustainable").json()
    assert second["source"] == "database"
