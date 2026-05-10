"""Tests for GET /api/lookup/{word}."""
import json

import main


def test_extract_json_value_ignores_surrounding_model_text():
    text = '```json\n{"pos": "noun", "cn": "测试"}\n```'
    assert main._extract_json_value(text, dict) == {"pos": "noun", "cn": "测试"}


def test_word_to_dict_tolerates_corrupt_context():
    word = main.DBWord(
        word="rough",
        pos="adj.",
        cn="粗糙的",
        en_definition="not smooth",
        context="not-json",
        example="The surface is rough.",
        image_path="/static/generated_images/missing.jpg",
    )
    data = main._word_to_dict(word)
    assert data["tags"] == []
    assert data["image_url"] is None


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


def test_lookup_translation_failure_uses_local_fallback(client, monkeypatch):
    monkeypatch.setattr(main, "fetch_dictionary", lambda w: {
        "meanings": [{
            "partOfSpeech": "noun",
            "definitions": [{
                "definition": "an alphabetical listing of items and their location",
                "example": "The index lists every topic.",
            }],
        }],
    })

    def _translate_down(*a, **kw):
        raise RuntimeError("Gemini timeout")

    monkeypatch.setattr(main, "ai_translate", _translate_down)
    monkeypatch.setattr(main, "ai_extract_topic_tags", lambda w, d, allowed: [])

    r = client.get("/api/lookup/index")
    assert r.status_code == 200
    assert r.json()["data"]["cn"] == "索引；指数"


def test_lookup_uses_free_translation_before_gemini(client, monkeypatch):
    monkeypatch.setattr(main, "fetch_dictionary", lambda w: {
        "meanings": [{
            "partOfSpeech": "noun",
            "definitions": [{
                "definition": "a compact electronic device",
                "example": "The device is easy to carry.",
            }],
        }],
    })
    monkeypatch.setattr(main, "fetch_free_translation", lambda w, d: "设备")
    monkeypatch.setattr(main, "ai_extract_topic_tags", lambda w, d, allowed: [])

    def _should_not_run(*a, **kw):
        raise AssertionError("Gemini translation should not run")

    monkeypatch.setattr(main, "ai_translate", _should_not_run)

    r = client.get("/api/lookup/device")
    assert r.status_code == 200
    assert r.json()["data"]["cn"] == "设备"


def test_lookup_repairs_cached_translation_failure(client, db_session):
    db_session.add(main.DBWord(
        word="look",
        pos="verb",
        cn="(translation failed)",
        en_definition="direct one's gaze toward someone or something",
        context=json.dumps([]),
        example="Look carefully at the question.",
    ))
    db_session.commit()

    r = client.get("/api/lookup/look")
    assert r.status_code == 200
    assert r.json()["data"]["cn"] == "看；查看"

    stored = db_session.query(main.DBWord).filter(main.DBWord.word == "look").first()
    assert stored.cn == "看；查看"


def test_lookup_repairs_cached_obscure_definition_and_missing_example(
    client, db_session
):
    db_session.add(main.DBWord(
        word="tree",
        pos="noun, verb",
        cn="tree：基于 Kruskal 树定理的快速生长函数。",
        en_definition="Fast growing function based on Kruskal's tree theorem.",
        context=json.dumps([]),
        example="",
    ))
    db_session.commit()

    r = client.get("/api/lookup/tree")
    assert r.status_code == 200
    data = r.json()["data"]
    assert data["cn"] == "树"
    assert data["en_definition"] == "a tall plant with a trunk, branches, and leaves"
    assert data["example"] == "Birds built a nest in the tree near the garden."
    assert data["tags"] == ["Nature"]


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


def test_lookup_full_failure_uses_offline_fallback(client, monkeypatch):
    """Both DictionaryAPI and ai_base_info fail -> offline fallback."""
    def _boom(*a, **kw):
        raise RuntimeError("AI down")

    monkeypatch.setattr(main, "fetch_dictionary", lambda w: None)
    monkeypatch.setattr(main, "ai_base_info", _boom)
    monkeypatch.setattr(main, "ai_extract_topic_tags", _boom)

    r = client.get("/api/lookup/look")
    assert r.status_code == 200
    data = r.json()["data"]
    assert data["cn"] == "看；查看"
    assert data["en_definition"] == "to direct your eyes toward someone or something"
    assert data["tags"] == ["Education"]


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


def test_generate_image_endpoint_uses_local_fallback_when_public_disabled(
    client, db_session, monkeypatch
):
    monkeypatch.setattr(main, "PUBLIC_IMAGE_API_ENABLED", False)
    db_session.add(main.DBWord(
        word="tree",
        pos="noun",
        cn="树",
        en_definition="a tall plant with a trunk, branches, and leaves",
        context=json.dumps(["Nature"]),
        example="Birds built a nest in the tree near the garden.",
    ))
    db_session.commit()

    r = client.post("/api/generate_image/tree")
    assert r.status_code == 200
    body = r.json()
    assert body["provider"] == "local"
    assert body["data"]["image_url"].startswith("data:image/svg+xml")


def test_generate_image_repairs_cached_word_before_prompt(
    client, db_session, monkeypatch
):
    monkeypatch.setattr(main, "PUBLIC_IMAGE_API_ENABLED", False)
    db_session.add(main.DBWord(
        word="tree",
        pos="noun, verb",
        cn="tree：基于 Kruskal 树定理的快速生长函数。",
        en_definition="Fast growing function based on Kruskal's tree theorem.",
        context=json.dumps([]),
        example="",
    ))
    db_session.commit()

    r = client.post("/api/generate_image/tree")
    assert r.status_code == 200
    data = r.json()["data"]
    assert data["cn"] == "树"
    assert data["en_definition"] == "a tall plant with a trunk, branches, and leaves"
    assert "tree" in data["image_url"]


def test_generate_image_stores_metadata(client, db_session, monkeypatch):
    """Verify provider, prompt, and created_at are saved to the DB."""
    monkeypatch.setattr(main, "PUBLIC_IMAGE_API_ENABLED", False)
    db_session.add(main.DBWord(
        word="tree",
        pos="noun",
        cn="树",
        en_definition="a tall plant with a trunk, branches, and leaves",
        context=json.dumps(["Nature"]),
        example="Example sentence.",
    ))
    db_session.commit()

    r = client.post("/api/generate_image/tree")
    assert r.status_code == 200

    stored = db_session.query(main.DBWord).filter(main.DBWord.word == "tree").first()
    assert stored.image_provider == "local"
    assert stored.image_prompt is not None
    assert "tree" in stored.image_prompt
    assert stored.image_created_at is not None
    assert "T" in stored.image_created_at  # ISO timestamp


def test_generate_image_size_parameter_affects_prompt(client, db_session, monkeypatch):
    """size query param should change the resolution in the prompt."""
    monkeypatch.setattr(main, "PUBLIC_IMAGE_API_ENABLED", False)
    db_session.add(main.DBWord(
        word="tree",
        pos="noun",
        cn="树",
        en_definition="a tall plant",
        context=json.dumps(["Nature"]),
        example="Example.",
    ))
    db_session.commit()

    r_small = client.post("/api/generate_image/tree?size=small")
    r_large = client.post("/api/generate_image/tree?size=large")
    assert r_small.status_code == 200
    assert r_large.status_code == 200

    small_prompt = db_session.query(main.DBWord).filter(
        main.DBWord.image_prompt.like("%512x512%")).first()
    large_prompt = db_session.query(main.DBWord).filter(
        main.DBWord.image_prompt.like("%1024x1024%")).first()
    # The last request's prompt is what's in the DB. It should contain
    # the large dimension since force=true wasn't used (so the second POST
    # overwrites).
    stored = db_session.query(main.DBWord).filter(main.DBWord.word == "tree").first()
    assert "1024x1024" in stored.image_prompt


def test_generate_image_force_regen(client, db_session, monkeypatch):
    """force=true should clear image_path and regenerate."""
    monkeypatch.setattr(main, "PUBLIC_IMAGE_API_ENABLED", False)
    db_session.add(main.DBWord(
        word="tree",
        pos="noun",
        cn="树",
        en_definition="a tall plant",
        context=json.dumps(["Nature"]),
        example="Example.",
        image_path="/static/generated_images/old-image.png",
        image_provider="gemini",
        image_prompt="old prompt",
        image_created_at="2025-01-01T00:00:00Z",
    ))
    db_session.commit()

    r = client.post("/api/generate_image/tree?force=true")
    assert r.status_code == 200

    stored = db_session.query(main.DBWord).filter(main.DBWord.word == "tree").first()
    # Should have been regenerated by local fallback, not the old gemini image.
    assert stored.image_provider == "local"
    assert stored.image_path.startswith("data:image/svg+xml")
    assert stored.image_prompt != "old prompt"


def test_generate_image_debug_mode_includes_extra_fields(client, db_session, monkeypatch):
    """DEBUG=true should include prompt and created_at in the response."""
    monkeypatch.setattr(main, "PUBLIC_IMAGE_API_ENABLED", False)
    monkeypatch.setattr(main, "DEBUG", True)
    db_session.add(main.DBWord(
        word="tree",
        pos="noun",
        cn="树",
        en_definition="a tall plant",
        context=json.dumps(["Nature"]),
        example="Example.",
    ))
    db_session.commit()

    r = client.post("/api/generate_image/tree")
    assert r.status_code == 200
    body = r.json()
    assert "prompt" in body
    assert "created_at" in body
    assert "T" in body["created_at"]


def test_generate_image_invalid_size_returns_400(client, db_session):
    db_session.add(main.DBWord(
        word="tree",
        pos="noun",
        cn="树",
        en_definition="a tall plant",
        context=json.dumps(["Nature"]),
        example="Example.",
    ))
    db_session.commit()

    r = client.post("/api/generate_image/tree?size=xlarge")
    assert r.status_code == 400


def test_generate_image_daily_quota_limits_each_ip(client, db_session, monkeypatch):
    monkeypatch.setattr(main, "IMAGE_DAILY_LIMIT_PER_IP", 1)
    monkeypatch.setattr(main, "PUBLIC_IMAGE_API_ENABLED", False)
    db_session.add(main.DBWord(
        word="tree",
        pos="noun",
        cn="树",
        en_definition="a tall plant",
        context=json.dumps(["Nature"]),
        example="Example.",
    ))
    db_session.commit()

    first = client.post("/api/generate_image/tree", headers={"x-forwarded-for": "203.0.113.10"})
    second = client.post("/api/generate_image/tree", headers={"x-forwarded-for": "203.0.113.10"})
    other_ip = client.post("/api/generate_image/tree", headers={"x-forwarded-for": "203.0.113.11"})

    assert first.status_code == 200
    assert second.status_code == 429
    assert "Daily AI image generation limit reached" in second.json()["detail"]
    assert other_ip.status_code == 200
