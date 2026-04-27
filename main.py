"""IELTS-Mate AI backend.

Single-file FastAPI app:
  - DictionaryAPI for English definitions/examples
  - Gemini for Chinese translation, topic tagging, and word suggestions
  - SQLite cache via SQLAlchemy

Each external dependency is wrapped in a small helper so tests can monkeypatch
them independently — see tests/conftest.py.
"""
import json
import os
import re
from datetime import date

import requests
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, create_engine, func
from sqlalchemy.orm import Session, declarative_base, sessionmaker

# ==========================================
# 0. Environment
# ==========================================
load_dotenv()

# Optional proxy — only applied when explicitly configured
_proxy = os.getenv("HTTP_PROXY_URL")
if _proxy:
    os.environ["HTTP_PROXY"] = _proxy
    os.environ["HTTPS_PROXY"] = _proxy

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TEXT_MODEL_ID = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
DB_URL = os.getenv("DATABASE_URL", "sqlite:///./ielts_mate.db")

# ==========================================
# 1. Database
# ==========================================
_connect_args = {"check_same_thread": False} if DB_URL.startswith("sqlite") else {}
engine = create_engine(DB_URL, connect_args=_connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class DBWord(Base):
    __tablename__ = "words"
    id = Column(Integer, primary_key=True, index=True)
    word = Column(String, unique=True, index=True)
    pos = Column(String)
    cn = Column(String)
    en_definition = Column(String)
    # JSON-encoded list of topic tags (subset of ALL_INTEREST_TAGS).
    # Stored as text so SQLite LIKE can match.
    context = Column(String, default="[]")
    example = Column(String)


class UserPreference(Base):
    __tablename__ = "user_preferences"
    id = Column(Integer, primary_key=True, index=True)
    selected_tags = Column(String, default="[]")
    last_daily_word_date = Column(String)
    last_daily_word_id = Column(Integer)


Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ==========================================
# 2. Gemini configuration (lazy)
# ==========================================
_genai_configured = False


def _ensure_genai_configured():
    global _genai_configured
    if _genai_configured:
        return
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and fill it in."
        )
    import google.generativeai as genai

    genai.configure(api_key=GEMINI_API_KEY, transport="rest")
    _genai_configured = True


def _get_text_model():
    _ensure_genai_configured()
    import google.generativeai as genai

    return genai.GenerativeModel(TEXT_MODEL_ID)


# ==========================================
# 3. External-call helpers (each individually mockable)
# ==========================================
ALL_INTEREST_TAGS = [
    "Architecture", "Esports", "Anime", "Music", "Sports", "Board Games",
    "Programming", "Baking", "Business", "Technology", "Travel", "Food",
    "Nature", "Art", "History", "Health", "Society", "Finance", "Law",
    "Education", "Media", "Environment",
]


def fetch_dictionary(word: str) -> dict | None:
    """Hit dictionaryapi.dev. Return first entry or None on any failure."""
    try:
        r = requests.get(
            f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}",
            timeout=5,
        )
        if r.status_code != 200:
            return None
        data = r.json()
        return data[0] if isinstance(data, list) and data else None
    except Exception:
        return None


def ai_translate(word: str, en_definition: str) -> str:
    model = _get_text_model()
    prompt = (
        f"Translate the English word '{word}' to Chinese. "
        f"Definition for context: '{en_definition}'. "
        f"Return only the Chinese translation, no quotes, no extra words."
    )
    return model.generate_content(prompt).text.strip()


def ai_generate_example(word: str) -> str:
    model = _get_text_model()
    prompt = (
        f"Provide one clear example sentence using the IELTS word '{word}'. "
        f"Return only the sentence."
    )
    return model.generate_content(prompt).text.strip().replace('"', "")


def ai_base_info(word: str) -> dict:
    """Used when DictionaryAPI fails. Returns {pos, cn, en_definition}."""
    model = _get_text_model()
    prompt = (
        f"For the English word '{word}', return a JSON object with three keys: "
        f'"pos" (part of speech), "cn" (Chinese translation), '
        f'"en_definition" (a short English definition). '
        f"Return only the JSON, no markdown."
    )
    text = model.generate_content(prompt).text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"AI base_info returned no JSON: {text!r}")
    return json.loads(match.group(0))


def ai_extract_topic_tags(
    word: str, en_definition: str, allowed_tags: list[str]
) -> list[str]:
    """Pick which of allowed_tags this word relates to. Returns subset (may be [])."""
    model = _get_text_model()
    prompt = (
        f'For the English word "{word}" (meaning: "{en_definition}"), pick which '
        f"of the following topic tags this word is most associated with. "
        f"Return a JSON array of 0–3 tags chosen ONLY from this list: "
        f"{json.dumps(allowed_tags)}. Return only the JSON array."
    )
    text = model.generate_content(prompt).text
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        tags = json.loads(match.group(0))
        return [t for t in tags if t in allowed_tags]
    except Exception:
        return []


def ai_suggest_word_for_tags(tags: list[str]) -> str:
    model = _get_text_model()
    topic = ", ".join(tags) if tags else "general academic English"
    prompt = (
        f"Suggest a single uncommon IELTS-level English word related to: {topic}. "
        f"Return only the lowercase word, no punctuation."
    )
    text = model.generate_content(prompt).text.strip().lower().replace(".", "")
    return text.split()[0] if text else ""


# ==========================================
# 4. Word generation pipeline
# ==========================================
def _word_to_dict(w: DBWord) -> dict:
    return {
        "id": w.id,
        "word": w.word,
        "pos": w.pos,
        "cn": w.cn,
        "en_definition": w.en_definition,
        "example": w.example,
        "tags": json.loads(w.context) if w.context else [],
    }


def generate_and_save_word(db: Session, word: str) -> DBWord | None:
    """Look up + persist a word. Returns the saved DBWord, or None on full failure."""
    word = word.lower().strip()
    if not word:
        return None

    pos: str = "n/a"
    cn: str = "n/a"
    en_definition: str = "n/a"
    example: str | None = None

    dict_data = fetch_dictionary(word)
    if dict_data:
        try:
            meanings = dict_data.get("meanings") or []
            pos = ", ".join(sorted({m["partOfSpeech"] for m in meanings}))
            first_def = meanings[0]["definitions"][0]
            en_definition = first_def["definition"]
            example = first_def.get("example")
        except (KeyError, IndexError):
            dict_data = None

    if not dict_data:
        try:
            base = ai_base_info(word)
            pos = base.get("pos", "n/a")
            cn = base.get("cn", "n/a")
            en_definition = base.get("en_definition", "n/a")
        except Exception as e:
            print(f"!!! ai_base_info failed for '{word}': {e}")
            return None

    if cn == "n/a":
        try:
            cn = ai_translate(word, en_definition)
        except Exception as e:
            print(f"!!! ai_translate failed for '{word}': {e}")
            cn = "(translation failed)"

    if not example:
        try:
            example = ai_generate_example(word)
        except Exception as e:
            print(f"!!! ai_generate_example failed for '{word}': {e}")
            example = ""

    try:
        tags = ai_extract_topic_tags(word, en_definition, ALL_INTEREST_TAGS)
    except Exception as e:
        print(f"!!! ai_extract_topic_tags failed for '{word}': {e}")
        tags = []

    new_word = DBWord(
        word=word,
        pos=pos,
        cn=cn,
        en_definition=en_definition,
        context=json.dumps(tags),
        example=example or "",
    )
    db.add(new_word)
    try:
        db.commit()
        db.refresh(new_word)
        return new_word
    except Exception:
        db.rollback()
        # Race / unique constraint — return whatever's now in the DB.
        return db.query(DBWord).filter(DBWord.word == word).first()


# ==========================================
# 5. FastAPI app + routes
# ==========================================
app = FastAPI(title="IELTS-Mate AI", version="1.0.0")


@app.get("/api/lookup/{word}")
def lookup_word(word: str, db: Session = Depends(get_db)):
    search_word = word.lower().strip()
    if not search_word:
        raise HTTPException(status_code=400, detail="Word is empty.")

    cached = db.query(DBWord).filter(DBWord.word == search_word).first()
    if cached:
        return {"source": "database", "data": _word_to_dict(cached)}

    new_word = generate_and_save_word(db, search_word)
    if not new_word:
        raise HTTPException(status_code=502, detail="Failed to generate word data.")
    return {"source": "generated", "data": _word_to_dict(new_word)}


class PreferenceRequest(BaseModel):
    selected_tags: list[str]


@app.get("/api/preferences")
def get_preferences(db: Session = Depends(get_db)):
    pref = db.query(UserPreference).first()
    if not pref:
        return {"selected_tags": []}
    return {"selected_tags": json.loads(pref.selected_tags or "[]")}


@app.post("/api/preferences")
def save_preferences(req: PreferenceRequest, db: Session = Depends(get_db)):
    pref = db.query(UserPreference).first()
    if pref:
        pref.selected_tags = json.dumps(req.selected_tags)
    else:
        pref = UserPreference(selected_tags=json.dumps(req.selected_tags))
        db.add(pref)
    # Changing tags invalidates the cached "today's word".
    pref.last_daily_word_date = None
    pref.last_daily_word_id = None
    db.commit()
    return {"selected_tags": req.selected_tags}


@app.get("/api/daily_word")
def daily_word(db: Session = Depends(get_db)):
    today = date.today().isoformat()
    pref = db.query(UserPreference).first()
    if not pref:
        pref = UserPreference(selected_tags="[]")
        db.add(pref)
        db.commit()
        db.refresh(pref)

    if pref.last_daily_word_date == today and pref.last_daily_word_id:
        cached = db.query(DBWord).filter(DBWord.id == pref.last_daily_word_id).first()
        if cached:
            return {"source": "cached", "data": _word_to_dict(cached)}

    selected_tags = json.loads(pref.selected_tags or "[]")
    word_obj: DBWord | None = None

    # 1. Try the DB first. If tags are selected, filter by them; otherwise pick
    # any random word that's already cached (avoids an AI call when we don't need one).
    q = db.query(DBWord)
    for tag in selected_tags:
        q = q.filter(DBWord.context.like(f"%{tag}%"))
    word_obj = q.order_by(func.random()).first()

    # 2. Ask AI for a fresh suggestion.
    if not word_obj:
        try:
            suggested = ai_suggest_word_for_tags(selected_tags)
            if suggested:
                existing = db.query(DBWord).filter(DBWord.word == suggested).first()
                word_obj = existing or generate_and_save_word(db, suggested)
        except Exception as e:
            print(f"!!! daily_word AI suggestion failed: {e}")

    # 3. Safety net: any random word in the DB.
    if not word_obj:
        word_obj = db.query(DBWord).order_by(func.random()).first()

    # 4. Last resort: only attempt fresh generation if API is configured.
    if not word_obj and GEMINI_API_KEY:
        word_obj = generate_and_save_word(db, "welcome")

    if not word_obj:
        raise HTTPException(
            status_code=503,
            detail=(
                "No word available. Configure GEMINI_API_KEY in .env or hit "
                "/api/lookup/<word> to seed the database."
            ),
        )

    pref.last_daily_word_date = today
    pref.last_daily_word_id = word_obj.id
    db.commit()
    return {"source": "fresh", "data": _word_to_dict(word_obj)}


@app.get("/api/pre-cache-tags")
def pre_cache_tags(db: Session = Depends(get_db)):
    """Generate one representative word per interest tag. Requires API key."""
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY not configured.",
        )

    generated = 0
    failed: list[str] = []
    for tag in ALL_INTEREST_TAGS:
        try:
            existing = db.query(DBWord).filter(DBWord.context.like(f"%{tag}%")).first()
            if existing:
                continue
            suggested = ai_suggest_word_for_tags([tag])
            if not suggested:
                failed.append(tag)
                continue
            if db.query(DBWord).filter(DBWord.word == suggested).first():
                continue
            saved = generate_and_save_word(db, suggested)
            if saved:
                generated += 1
            else:
                failed.append(tag)
        except Exception as e:
            print(f"!!! pre_cache failed for '{tag}': {e}")
            failed.append(tag)
    return {"generated": generated, "failed": failed}


# ==========================================
# 6. Static files
# ==========================================
# Mount the frontend at /static (NOT /) so source files, .env, and the SQLite
# database in the project root are NOT served over HTTP.
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
