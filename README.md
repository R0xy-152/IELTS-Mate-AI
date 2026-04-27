# IELTS-Mate AI

A small IELTS vocabulary helper that pairs the free
[DictionaryAPI](https://dictionaryapi.dev/) with
[Google Gemini](https://ai.google.dev/) to deliver:

- bilingual word definitions,
- topic-aware "word of the day" recommendations driven by interest tags,
- and a cached lookup history backed by SQLite.

> **Honesty note.** Earlier versions of this README claimed multimodal video and
> 4K image generation. That was not implemented and has been removed. This
> project is text-only — see *Roadmap* below for what's intentionally out of
> scope.

---

## Features

| | |
|---|---|
| 🔍 **Hybrid lookup** | DictionaryAPI for English POS / definition / example, Gemini for the Chinese translation |
| 🏷️ **Topic tagging** | Gemini classifies each word into 0–3 of 22 interest categories (Tech, Sports, Music…) |
| 📅 **Daily word** | Picks a fresh word matching your saved interests, then caches it for the rest of the day |
| 🧱 **SQLite cache** | Every looked-up word is persisted; subsequent requests hit the DB and never call out |
| 🧪 **Offline tests** | All external calls are isolated; full test suite runs without an API key or network |

---

## Quick start

### 1. Install

```bash
git clone https://github.com/R0xy-152/IELTS-Mate-AI.git
cd IELTS-Mate-AI
python -m venv .venv
source .venv/bin/activate    # macOS / Linux
# .venv\Scripts\activate     # Windows PowerShell
pip install -r requirements.txt
```

### 2. Configure

Copy the env template and fill in your Gemini key:

```bash
cp .env.example .env
```

Then edit `.env`:

```env
GEMINI_API_KEY=your_real_key_here
```

A free key is available at <https://aistudio.google.com/apikey>.

> **Behind a corporate / restricted network?** Set `HTTP_PROXY_URL` in `.env`
> to your local proxy (e.g. `http://127.0.0.1:7897` for Clash Verge).

### 3. Run the server

```bash
uvicorn main:app --reload --port 8000
```

Open <http://127.0.0.1:8000>.

---

## Running tests (no API key required)

The test suite mocks every external dependency, so it runs fully offline:

```bash
pytest
```

Expected output:

```
============================= 18 passed in 1.01s ==============================
```

If you want to also verify your real Gemini key is reachable, add an
integration test (or just hit the running server with `curl`):

```bash
curl http://127.0.0.1:8000/api/lookup/sustainability | python -m json.tool
```

---

## API

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/api/lookup/{word}`  | Look up a word (DB cache → DictionaryAPI → Gemini) |
| `GET`  | `/api/daily_word`     | Today's word, biased by saved interest tags |
| `GET`  | `/api/preferences`    | Read selected interest tags |
| `POST` | `/api/preferences`    | Update interest tags (clears the daily-word cache) |
| `GET`  | `/api/pre-cache-tags` | One-shot warm-up: generate one word per interest tag |
| `GET`  | `/`                   | Serves the static frontend (`static/index.html`) |

### Response shape (`/api/lookup/{word}`)

```json
{
  "source": "database" | "generated",
  "data": {
    "id": 1,
    "word": "sustainability",
    "pos": "noun",
    "cn": "可持续性",
    "en_definition": "the ability to be maintained at a certain rate",
    "example": "Long-term sustainability is the goal.",
    "tags": ["Environment"]
  }
}
```

---

## Project layout

```
.
├── main.py                # All backend code (single file by design)
├── static/
│   └── index.html         # Frontend SPA
├── tests/
│   ├── conftest.py        # In-memory DB + auto-stubbed AI/HTTP fixtures
│   ├── test_lookup.py
│   ├── test_daily_word.py
│   └── test_preferences.py
├── .env.example           # Env template — copy to .env and fill in
├── requirements.txt
├── pytest.ini
├── README.md              # ← this file
└── CLAUDE.md              # Project guide for Claude Code sessions
```

---

## Configuration reference

All configuration is read from environment variables (loaded via `.env` if present).

| Variable | Required | Default | Notes |
|---|---|---|---|
| `GEMINI_API_KEY`  | ✅ for the server | — | Not required for `pytest` |
| `GEMINI_MODEL`    | — | `gemini-2.5-flash` | Override only if you have access to a newer model |
| `HTTP_PROXY_URL`  | — | — | Set this if you need a local proxy to reach Google |
| `DATABASE_URL`    | — | `sqlite:///./ielts_mate.db` | Any SQLAlchemy URL works |

---

## Security notes

- The frontend is mounted at `/static/`; the project root is **not** served.
  This means `.env`, `*.db`, and `main.py` cannot be downloaded over HTTP.
  There is a regression test (`test_static_mount_does_not_expose_dotenv`) to
  protect this.
- Never commit your real `.env`. The repo's `.gitignore` already excludes it.
- If you suspect a key has leaked, rotate it immediately at
  <https://aistudio.google.com/apikey>.

---

## Roadmap / known limitations

These are intentionally **out of scope** in this version:

- ❌ Multimodal output (no image / video generation; the "AI Chat" tab is removed)
- ❌ User accounts (preferences are global to the database)
- ❌ Authentication / rate limiting (don't expose this to the public internet as-is)
- ❌ Real-time speech (no STT / TTS integration)

Reasonable next steps if you want to extend it:

- Add an `Imagen` or `gemini-2.5-flash-image-preview` integration for real
  illustrations, gated by a `IMAGES_ENABLED` flag.
- Add a `User` model and a per-user `preferences` row.
- Add a `pytest -m integration` group that talks to the real Gemini API.

---

## Tech stack

FastAPI · SQLAlchemy · SQLite · Google Gemini API · DictionaryAPI.dev · Vanilla JS

## License

MIT (educational project).
