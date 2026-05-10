"""Tests for IELTS Speaking Examiner endpoints."""

import main


def test_speaking_start_creates_session_and_returns_examiner_message(client, db_session):
    r = client.post("/api/speaking/start")
    assert r.status_code == 200
    body = r.json()
    assert "session" in body
    assert body["session"]["status"] == "active"
    assert body["session"]["topic"] in main.SPEAKING_TOPICS
    assert len(body["messages"]) == 1
    assert body["messages"][0]["role"] == "examiner"
    assert "Examiner question #1" in body["messages"][0]["content"]


def test_speaking_start_503_when_no_api_key(client, monkeypatch):
    monkeypatch.setattr(main, "GEMINI_API_KEY", None)
    r = client.post("/api/speaking/start")
    assert r.status_code == 503


def test_speaking_respond_saves_and_replies(client, db_session):
    # Start a session first
    r = client.post("/api/speaking/start")
    body = r.json()
    session_id = body["session"]["id"]

    # Send a response
    r = client.post(f"/api/speaking/{session_id}/respond", json={"message": "I live in Beijing."})
    assert r.status_code == 200
    body = r.json()
    assert body["session"]["id"] == session_id

    # Should have 3 messages: examiner #1, user, examiner #2
    msgs = body["messages"]
    assert len(msgs) == 3
    assert msgs[0]["role"] == "examiner"
    assert msgs[1]["role"] == "user"
    assert msgs[1]["content"] == "I live in Beijing."
    assert msgs[2]["role"] == "examiner"
    assert "Examiner question #2" in msgs[2]["content"]


def test_speaking_respond_404_for_invalid_session(client):
    r = client.post("/api/speaking/9999/respond", json={"message": "Hello"})
    assert r.status_code == 404


def test_speaking_respond_400_for_empty_message(client, db_session):
    r = client.post("/api/speaking/start")
    session_id = r.json()["session"]["id"]

    r = client.post(f"/api/speaking/{session_id}/respond", json={"message": ""})
    assert r.status_code == 400


def test_speaking_respond_400_for_completed_session(client, db_session):
    r = client.post("/api/speaking/start")
    session_id = r.json()["session"]["id"]

    # Build up enough messages to evaluate
    client.post(f"/api/speaking/{session_id}/respond", json={"message": "Hello"})
    client.post(f"/api/speaking/{session_id}/respond", json={"message": "I'm fine thanks"})

    # Evaluate
    client.post(f"/api/speaking/{session_id}/evaluate")

    # Responding again should fail
    r = client.post(f"/api/speaking/{session_id}/respond", json={"message": "Too late"})
    assert r.status_code == 400


def test_speaking_evaluate_returns_scores(client, db_session):
    r = client.post("/api/speaking/start")
    session_id = r.json()["session"]["id"]

    # Add two user messages (4 total exchanges incl. examiner)
    client.post(f"/api/speaking/{session_id}/respond", json={"message": "Hello"})
    client.post(f"/api/speaking/{session_id}/respond", json={"message": "I enjoy reading books."})

    r = client.post(f"/api/speaking/{session_id}/evaluate")
    assert r.status_code == 200
    body = r.json()
    assert body["session"]["status"] == "completed"
    assert body["session"]["evaluation"] is not None
    evaluation = body["session"]["evaluation"]
    assert evaluation["overall_band"] == 6.0
    assert evaluation["fluency_coherence"]["score"] == 6.0
    assert evaluation["lexical_resource"]["score"] == 5.5
    assert evaluation["grammatical_range"]["score"] == 6.0
    assert len(evaluation["strengths"]) > 0
    assert len(evaluation["improvements"]) > 0
    assert len(evaluation["notable_vocabulary"]) > 0


def test_speaking_evaluate_400_for_too_few_messages(client, db_session):
    r = client.post("/api/speaking/start")
    session_id = r.json()["session"]["id"]

    # Only 2 messages (examiner opening + 0 user responses = insufficient)
    r = client.post(f"/api/speaking/{session_id}/evaluate")
    assert r.status_code == 400


def test_speaking_evaluate_400_for_already_completed(client, db_session):
    r = client.post("/api/speaking/start")
    session_id = r.json()["session"]["id"]

    client.post(f"/api/speaking/{session_id}/respond", json={"message": "Hello"})
    client.post(f"/api/speaking/{session_id}/respond", json={"message": "Books are great"})
    client.post(f"/api/speaking/{session_id}/evaluate")

    # Second evaluation should fail
    r = client.post(f"/api/speaking/{session_id}/evaluate")
    assert r.status_code == 400


def test_speaking_full_session_flow(client, db_session):
    """End-to-end: start, multiple exchanges, evaluate."""
    # Start
    r = client.post("/api/speaking/start")
    assert r.status_code == 200
    session_id = r.json()["session"]["id"]
    topic = r.json()["session"]["topic"]
    assert topic in main.SPEAKING_TOPICS

    # Multiple exchanges
    for msg in ["I am a student.", "I study computer science.", "I like programming."]:
        r = client.post(f"/api/speaking/{session_id}/respond", json={"message": msg})
        assert r.status_code == 200

    # Evaluate
    r = client.post(f"/api/speaking/{session_id}/evaluate")
    assert r.status_code == 200
    evaluation = r.json()["session"]["evaluation"]
    assert evaluation is not None

    # Verify DB state
    session = db_session.query(main.DBSpeakingSession).filter(
        main.DBSpeakingSession.id == session_id
    ).first()
    assert session.status == "completed"
    assert session.completed_at is not None
    assert session.evaluation is not None

    messages = (
        db_session.query(main.DBSpeakingMessage)
        .filter(main.DBSpeakingMessage.session_id == session_id)
        .order_by(main.DBSpeakingMessage.id)
        .all()
    )
    assert len(messages) >= 7  # opener + 3 user + 3 examiner responses
