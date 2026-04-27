# CLAUDE.md

Project guide for Claude Code sessions working in this repository.

## What this project is

A small, single-developer FastAPI service that:

1. Looks up English words by combining DictionaryAPI (free, no key) and Gemini
   (Chinese translation, topic tagging, and example fallback).
2. Persists every lookup in a SQLite cache so subsequent requests are free.
3. Serves a static SPA at `/` for browsing.

This is deliberately a **single-file backend** (`main.py`). Don't split it
into a package layout unless you have a real reason — the file is small.

## Tech stack

- Python 3.10+ (uses `X | None` union syntax)
- FastAPI + SQLAlchemy 2.x + SQLite
- `google-generativeai` for Gemini (model `gemini-2.5-flash` by default)
- Vanilla JS frontend in `static/index.html` — no build step

## Common commands

```bash
# First-time setup
python -m venv .venv
source .venv/bin/activate          # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env               # then fill GEMINI_API_KEY

# Run the server (needs a real key)
uvicorn main:app --reload --port 8000

# Run the test suite (NO key needed — every external call is stubbed)
pytest
```

## Layout

```
main.py                # All backend logic
static/index.html      # Frontend SPA
tests/
  conftest.py          # In-memory SQLite + auto-stubbed externals
  test_lookup.py
  test_daily_word.py
  test_preferences.py
.env.example           # Template for .env
requirements.txt
pytest.ini
```

## Architectural conventions

### 1. External calls live in named helpers

Every outbound call is wrapped in a single-purpose function so tests can
monkeypatch it independently:

| Helper | Calls |
|---|---|
| `fetch_dictionary(word)` | `dictionaryapi.dev` |
| `ai_translate(word, en_def)` | Gemini |
| `ai_generate_example(word)` | Gemini |
| `ai_base_info(word)` | Gemini (DictAPI fallback path) |
| `ai_extract_topic_tags(word, en_def, allowed)` | Gemini |
| `ai_suggest_word_for_tags(tags)` | Gemini |

**Don't call `genai.GenerativeModel(...)` directly inside route handlers** —
add a new `ai_*` helper instead so tests stay clean.

### 2. AI configuration is lazy

`_ensure_genai_configured()` runs only on first use. This is what lets
`pytest` import `main` without a `GEMINI_API_KEY` set.

### 3. DB schema

```
words(id, word, pos, cn, en_definition, context, example)
user_preferences(id, selected_tags, last_daily_word_date, last_daily_word_id)
```

`words.context` and `user_preferences.selected_tags` are both
**JSON-encoded lists of topic tags** stored as TEXT. Tag matching is
`LIKE '%TagName%'` against the JSON string. Don't change this without also
updating both write sites and `/api/daily_word`.

`ALL_INTEREST_TAGS` in `main.py` is the canonical tag vocabulary. The
frontend's `data-tag` attributes must match these exactly (English names).

### 4. Static-file isolation

```python
app.mount("/static", StaticFiles(directory="static"))
@app.get("/") -> serves static/index.html
```

The mount is **deliberately not at `/`**. Mounting at `/` would expose
`.env`, the SQLite file, and source code. There's a regression test
(`test_static_mount_does_not_expose_dotenv`) — keep it green.

### 5. Tests use FastAPI dependency override

`conftest.py` overrides `get_db` with an in-memory SQLite engine using
`StaticPool` so all sessions share the same connection. The `stub_externals`
fixture is `autouse=True` and replaces every `ai_*` and `fetch_dictionary`
function with deterministic defaults; tests override individual stubs as
needed.

## Environment variables

All read at import time in `main.py`:

| Variable | Required | Default |
|---|---|---|
| `GEMINI_API_KEY` | for server only | — |
| `GEMINI_MODEL` | no | `gemini-2.5-flash` |
| `HTTP_PROXY_URL` | no | unset |
| `DATABASE_URL` | no | `sqlite:///./ielts_mate.db` |

## Don't do these

- ❌ Don't reintroduce `video_url` / `image_url` columns or fields in
  responses without an actual generation provider. The previous version
  hardcoded a Big Buck Bunny demo video; that's why they were removed.
- ❌ Don't mount static files at `/`.
- ❌ Don't hardcode the proxy or API key in source. Always read from env.
- ❌ Don't claim multimodal / Veo / Nano-Banana in the README unless they're
  actually wired up and tested.
- ❌ Don't commit `.env`, `*.db`, or `__pycache__/`. They're git-ignored —
  keep it that way.

## When adding a feature

1. If it talks to an external service, add a new `*_helper(...)` at module top.
2. Add a default stub for it in `tests/conftest.py::stub_externals`.
3. Write at least one happy-path and one failure-path test.
4. Update README API table if you add a route.
5. Update `requirements.txt` if you add a dependency.

## Useful references

- Gemini quickstart: <https://ai.google.dev/gemini-api/docs/quickstart>
- DictionaryAPI: <https://dictionaryapi.dev/>
- FastAPI testing: <https://fastapi.tiangolo.com/tutorial/testing/>
