"""IELTS-Mate AI backend.

Single-file FastAPI app:
  - DictionaryAPI for English definitions/examples
  - Gemini for Chinese translation, topic tagging, and word suggestions
  - SQLite cache via SQLAlchemy

Each external dependency is wrapped in a small helper so tests can monkeypatch
them independently — see tests/conftest.py.
"""
from __future__ import annotations

import json
import os
import re
import hashlib
import base64
import time
from urllib.parse import quote
from datetime import date

import requests
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, create_engine, func, or_, text
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
IMAGE_MODEL_ID = os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")
DB_URL = os.getenv("DATABASE_URL", "sqlite:///./ielts_mate.db")
GEMINI_TIMEOUT_SECONDS = float(os.getenv("GEMINI_TIMEOUT_SECONDS", "12"))
GEMINI_RETRIES = int(os.getenv("GEMINI_RETRIES", "2"))
TRANSLATION_TIMEOUT_SECONDS = float(os.getenv("TRANSLATION_TIMEOUT_SECONDS", "4"))
IMAGE_PROXY_URL = os.getenv("IMAGE_PROXY_URL")
PUBLIC_IMAGE_API_ENABLED = os.getenv("PUBLIC_IMAGE_API_ENABLED", "true").lower() != "false"
IMAGE_TIMEOUT_SECONDS = float(os.getenv("IMAGE_TIMEOUT_SECONDS", "20"))
IMAGE_DAILY_LIMIT_PER_IP = int(os.getenv("IMAGE_DAILY_LIMIT_PER_IP", "1"))
RATE_LIMIT_SALT = os.getenv("RATE_LIMIT_SALT", "ielts-mate-ai")
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "").split(",")
    if origin.strip()
]
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

SIZE_MAP = {
    "small": 512,
    "medium": 768,
    "large": 1024,
}

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
    image_path = Column(String, nullable=True)
    image_provider = Column(String, nullable=True)
    image_prompt = Column(String, nullable=True)
    image_created_at = Column(String, nullable=True)


class UserPreference(Base):
    __tablename__ = "user_preferences"
    id = Column(Integer, primary_key=True, index=True)
    selected_tags = Column(String, default="[]")
    last_daily_word_date = Column(String)
    last_daily_word_id = Column(Integer)


class DBSpeakingSession(Base):
    __tablename__ = "speaking_sessions"
    id = Column(Integer, primary_key=True, index=True)
    topic = Column(String)
    status = Column(String, default="active")  # "active" | "completed"
    created_at = Column(String)
    completed_at = Column(String, nullable=True)
    evaluation = Column(String, nullable=True)  # JSON with scores


class DBSpeakingMessage(Base):
    __tablename__ = "speaking_messages"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, index=True)
    role = Column(String)  # "examiner" | "user"
    content = Column(String)
    created_at = Column(String)


class DBImageGenerationQuota(Base):
    __tablename__ = "image_generation_quota"
    id = Column(Integer, primary_key=True, index=True)
    ip_hash = Column(String, index=True)
    quota_date = Column(String, index=True)
    used_count = Column(Integer, default=0)


Base.metadata.create_all(bind=engine)

# ── Lightweight schema migration for image metadata columns ──
with engine.connect() as _conn:
    _existing_cols = {row[0] for row in _conn.execute(text("SELECT name FROM pragma_table_info('words')"))}
    for _col_name, _col_type in [
        ("image_provider", "VARCHAR"),
        ("image_prompt", "VARCHAR"),
        ("image_created_at", "VARCHAR"),
    ]:
        if _col_name not in _existing_cols:
            _conn.execute(text(f"ALTER TABLE words ADD COLUMN {_col_name} {_col_type}"))
    _conn.commit()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _client_ip(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for", "")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def _ip_hash(ip: str) -> str:
    return hashlib.sha256(f"{RATE_LIMIT_SALT}:{ip}".encode("utf-8")).hexdigest()


def _check_image_quota(db: Session, request: Request) -> DBImageGenerationQuota:
    if IMAGE_DAILY_LIMIT_PER_IP <= 0:
        return DBImageGenerationQuota(ip_hash="disabled", quota_date=str(date.today()), used_count=0)

    today = str(date.today())
    ip_hash = _ip_hash(_client_ip(request))
    quota = (
        db.query(DBImageGenerationQuota)
        .filter(
            DBImageGenerationQuota.ip_hash == ip_hash,
            DBImageGenerationQuota.quota_date == today,
        )
        .first()
    )
    if not quota:
        quota = DBImageGenerationQuota(ip_hash=ip_hash, quota_date=today, used_count=0)
        db.add(quota)
        db.flush()

    if quota.used_count >= IMAGE_DAILY_LIMIT_PER_IP:
        raise HTTPException(
            status_code=429,
            detail="Daily AI image generation limit reached. Try again tomorrow.",
        )
    return quota


def _consume_image_quota(db: Session, quota: DBImageGenerationQuota) -> None:
    if IMAGE_DAILY_LIMIT_PER_IP <= 0:
        return
    quota.used_count += 1
    db.add(quota)


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

DEFAULT_INTEREST_TAGS = ["Programming", "Technology", "Finance"]

FALLBACK_WORDS = [
    {
        "word": "algorithm",
        "pos": "noun",
        "cn": "算法",
        "en_definition": "a step-by-step procedure for solving a problem",
        "example": "A fair algorithm should be transparent and accountable.",
        "tags": ["Programming", "Technology"],
    },
    {
        "word": "innovation",
        "pos": "noun",
        "cn": "创新",
        "en_definition": "the introduction of new ideas, methods, or products",
        "example": "Technological innovation can reshape entire industries.",
        "tags": ["Technology", "Business"],
    },
    {
        "word": "liquidity",
        "pos": "noun",
        "cn": "流动性",
        "en_definition": "the ease with which an asset can be converted into cash",
        "example": "Strong liquidity helps companies handle unexpected expenses.",
        "tags": ["Finance", "Business"],
    },
]

LOCAL_TRANSLATIONS = {
    "algorithm": "算法",
    "analysis": "分析",
    "finance": "金融",
    "index": "索引；指数",
    "innovation": "创新",
    "liquidity": "流动性",
    "look": "看；查看",
    "programming": "编程",
    "sustainability": "可持续性",
    "technology": "技术；科技",
    "tree": "树",
    "welcome": "欢迎",
}

LOCAL_WORD_INFO = {
    "day": {
        "pos": "noun",
        "cn": "一天；白天",
        "en_definition": "a period of twenty-four hours",
        "example": "A productive day often starts with a clear plan.",
        "tags": ["Education"],
    },
    "index": {
        "pos": "noun",
        "cn": "索引；指数",
        "en_definition": "an alphabetical listing of items and their location",
        "example": "The index lists every important term in the book.",
        "tags": ["Education", "Finance"],
    },
    "look": {
        "pos": "verb",
        "cn": "看；查看",
        "en_definition": "to direct your eyes toward someone or something",
        "example": "Look carefully at the question before you answer.",
        "tags": ["Education"],
    },
    "test": {
        "pos": "noun, verb",
        "cn": "测试；考试",
        "en_definition": "a way of measuring knowledge, ability, or quality",
        "example": "The teacher designed a short test for the class.",
        "tags": ["Education"],
    },
    "tree": {
        "pos": "noun",
        "cn": "树",
        "en_definition": "a tall plant with a trunk, branches, and leaves",
        "example": "Birds built a nest in the tree near the garden.",
        "tags": ["Nature"],
    },
}

CN_UNAVAILABLE = "暂无中文释义"
CN_FAILURE_MARKERS = {"(translation failed)", "translation failed", "n/a", ""}


def _extract_json_value(text: str, expected_type: type) -> object:
    """Extract the first JSON value of the expected type from model output."""
    decoder = json.JSONDecoder()
    starts = "{" if expected_type is dict else "["
    for i, char in enumerate(text):
        if char != starts:
            continue
        try:
            value, _ = decoder.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, expected_type):
            return value
    raise ValueError(f"No {expected_type.__name__} JSON value found: {text!r}")


def _string_list_from_json(raw: str | None) -> list[str]:
    """Parse a JSON string list defensively, dropping non-string values."""
    if not raw:
        return []
    try:
        values = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(values, list):
        return []
    return [v for v in values if isinstance(v, str)]


def _dedupe_strings(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _preference_tags(raw: str | None) -> list[str]:
    return _string_list_from_json(raw) or DEFAULT_INTEREST_TAGS.copy()


def _as_text(value: object, default: str = "n/a") -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def _cn_needs_repair(cn: str | None) -> bool:
    return (cn or "").strip().lower() in CN_FAILURE_MARKERS


def _fallback_translation(word: str, en_definition: str = "") -> str:
    normalized = word.lower().strip()
    if normalized in LOCAL_TRANSLATIONS:
        return LOCAL_TRANSLATIONS[normalized]
    if "cash" in en_definition.lower() or "asset" in en_definition.lower():
        return "金融相关词汇"
    if "computer" in en_definition.lower() or "software" in en_definition.lower():
        return "科技相关词汇"
    return CN_UNAVAILABLE


def _fallback_example(word: str, en_definition: str = "") -> str:
    normalized = word.lower().strip()
    if normalized in LOCAL_WORD_INFO:
        return LOCAL_WORD_INFO[normalized]["example"]
    if en_definition and en_definition != "Definition unavailable while offline.":
        return f"The word '{normalized}' is useful when discussing {en_definition.rstrip('.').lower()}."
    return f"Please try looking up '{normalized}' again when the network is stable."


def _offline_base_info(word: str) -> dict:
    normalized = word.lower().strip()
    if normalized in LOCAL_WORD_INFO:
        return LOCAL_WORD_INFO[normalized]
    return {
        "pos": "n/a",
        "cn": _fallback_translation(normalized),
        "en_definition": "Definition unavailable while offline.",
        "example": f"Please try looking up '{normalized}' again when the network is stable.",
        "tags": [],
    }


def _placeholder_image_url() -> str | None:
    image_path = os.path.join(
        os.path.dirname(__file__),
        "static",
        "generated_images",
        "placeholder.jpg",
    )
    return "/static/generated_images/placeholder.jpg" if os.path.exists(image_path) else None


def _valid_image_url(image_url: str | None) -> str | None:
    if not image_url:
        return None
    if not image_url.startswith("/static/"):
        return image_url
    local_path = os.path.join(
        os.path.dirname(__file__),
        "static",
        image_url.removeprefix("/static/"),
    )
    return image_url if os.path.exists(local_path) else None


def _build_image_prompt(word: str, en_definition: str, tags: list[str], size: str = "small") -> str:
    dims = SIZE_MAP.get(size, 512)
    topic = ", ".join(tags) if tags else "IELTS vocabulary"
    return (
        f"Educational IELTS vocabulary illustration for the English word '{word}'. "
        f"Meaning: {en_definition}. Topic: {topic}. "
        f"Square {dims}x{dims} composition. "
        "Clear single concept, realistic editorial style, bright natural lighting, "
        "no text, no labels, no watermark."
    )


def _local_svg_image_url(word: str, en_definition: str, size: str = "small") -> str:
    dims = SIZE_MAP.get(size, 512)
    safe_word = re.sub(r"[^a-zA-Z0-9 -]", "", word)[:32] or "word"
    safe_definition = re.sub(r"[<>&]", "", en_definition)[:90] or "IELTS vocabulary"
    # Scale coordinates proportionally from the 1024×1024 reference design.
    scale = dims / 1024
    def s(val: float) -> int:
        return round(val * scale)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{dims}" height="{dims}" viewBox="0 0 {dims} {dims}">
<defs>
<linearGradient id="bg" x1="0" x2="1" y1="0" y2="1">
<stop offset="0" stop-color="#e8f1ff"/>
<stop offset="1" stop-color="#f7fbf0"/>
</linearGradient>
</defs>
<rect width="{dims}" height="{dims}" fill="url(#bg)"/>
<circle cx="{s(772)}" cy="{s(238)}" r="{s(132)}" fill="#0071e3" opacity="0.12"/>
<circle cx="{s(266)}" cy="{s(770)}" r="{s(180)}" fill="#34c759" opacity="0.13"/>
<rect x="{s(142)}" y="{s(278)}" width="{s(740)}" height="{s(468)}" rx="{s(48)}" fill="white" opacity="0.88"/>
<text x="{dims//2}" y="{s(466)}" text-anchor="middle" font-family="Arial, sans-serif" font-size="{s(96)}" font-weight="700" fill="#1d1d1f">{safe_word}</text>
<text x="{dims//2}" y="{s(552)}" text-anchor="middle" font-family="Arial, sans-serif" font-size="{s(34)}" fill="#515154">{safe_definition}</text>
<path d="M{s(352)} {s(650)}h{s(320)}" stroke="#0071e3" stroke-width="{s(14)}" stroke-linecap="round"/>
</svg>"""
    return "data:image/svg+xml;charset=utf-8," + quote(svg)


def _public_image_url(prompt: str, size: str = "small") -> str:
    dims = SIZE_MAP.get(size, 512)
    seed = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8], 16)
    return (
        "https://image.pollinations.ai/prompt/"
        f"{quote(prompt)}?width={dims}&height={dims}&model=flux&nologo=true&seed={seed}"
    )


def _save_generated_image(word: str, image_bytes: bytes, mime_type: str) -> str:
    ext = "jpg" if mime_type == "image/jpeg" else "png"
    digest = hashlib.sha256(image_bytes).hexdigest()[:16]
    safe_word = re.sub(r"[^a-z0-9-]", "-", word.lower()).strip("-") or "word"
    output_dir = os.path.join(os.path.dirname(__file__), "static", "generated_images")
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{safe_word}-{digest}.{ext}"
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "wb") as f:
        f.write(image_bytes)
    return f"/static/generated_images/{filename}"


def _gemini_image_url(prompt: str, word: str) -> str | None:
    if not GEMINI_API_KEY:
        return None
    try:
        r = requests.post(
            (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"{IMAGE_MODEL_ID}:generateContent"
            ),
            params={"key": GEMINI_API_KEY},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
            },
            timeout=IMAGE_TIMEOUT_SECONDS,
        )
        if r.status_code != 200:
            print(f"!!! Gemini image generation failed: status {r.status_code}")
            return None
        data = r.json()
        for candidate in data.get("candidates", []):
            for part in candidate.get("content", {}).get("parts", []):
                inline_data = part.get("inlineData") or part.get("inline_data")
                if not inline_data:
                    continue
                encoded = inline_data.get("data")
                mime_type = inline_data.get("mimeType") or inline_data.get("mime_type") or "image/png"
                if encoded and mime_type.startswith("image/"):
                    return _save_generated_image(
                        word,
                        base64.b64decode(encoded),
                        mime_type,
                    )
    except Exception as e:
        print(f"!!! Gemini image generation failed for '{word}': {e}")
    return None


def generate_image_url(word_obj: DBWord, size: str = "small") -> dict:
    tags = _string_list_from_json(word_obj.context)
    prompt = _build_image_prompt(word_obj.word, word_obj.en_definition or "", tags, size)
    created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def _result(url: str, provider: str) -> dict:
        return {"image_url": url, "provider": provider, "prompt": prompt, "created_at": created_at}

    if IMAGE_PROXY_URL:
        try:
            r = requests.post(
                IMAGE_PROXY_URL,
                json={
                    "word": word_obj.word,
                    "definition": word_obj.en_definition,
                    "tags": tags,
                    "prompt": prompt,
                    "size": size,
                },
                timeout=IMAGE_TIMEOUT_SECONDS,
            )
            if r.status_code == 200:
                data = r.json()
                image_url = data.get("image_url") or data.get("url")
                if image_url:
                    return _result(image_url, "proxy")
        except Exception as e:
            print(f"!!! image proxy failed for '{word_obj.word}': {e}")

    gemini_url = _gemini_image_url(prompt, word_obj.word)
    if gemini_url:
        return _result(gemini_url, "gemini")

    if PUBLIC_IMAGE_API_ENABLED:
        return _result(_public_image_url(prompt, size), "pollinations")

    return _result(_local_svg_image_url(word_obj.word, word_obj.en_definition or "", size), "local")


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _context_has_tag_clause(tag: str):
    json_tag = json.dumps(tag)
    return DBWord.context.like(f"%{_escape_like(json_tag)}%", escape="\\")


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


def fetch_free_translation(word: str, en_definition: str = "") -> str | None:
    """Use a no-key translation endpoint as a best-effort Chinese fallback."""
    text = f"{word}: {en_definition}" if en_definition else word
    try:
        r = requests.get(
            "https://api.mymemory.translated.net/get",
            params={"q": text, "langpair": "en|zh-CN"},
            timeout=TRANSLATION_TIMEOUT_SECONDS,
        )
        if r.status_code != 200:
            return None
        data = r.json()
        translated = (
            data.get("responseData", {})
            .get("translatedText", "")
            .strip()
        )
        if not translated:
            return None
        if translated.lower() == text.lower():
            return None
        return translated[:120]
    except Exception:
        return None


def _generate_text(prompt: str) -> str:
    model = _get_text_model()
    last_error: Exception | None = None
    for attempt in range(GEMINI_RETRIES + 1):
        try:
            response = model.generate_content(
                prompt,
                request_options={"timeout": GEMINI_TIMEOUT_SECONDS},
            )
            return (response.text or "").strip()
        except Exception as e:
            last_error = e
            if attempt >= GEMINI_RETRIES:
                break
            time.sleep(0.4 * (2 ** attempt))
    raise RuntimeError(f"Gemini request failed after retries: {last_error}")


def ai_translate(word: str, en_definition: str) -> str:
    prompt = (
        f"Translate the English word '{word}' to Chinese. "
        f"Definition for context: '{en_definition}'. "
        f"Return only the Chinese translation, no quotes, no extra words."
    )
    return _generate_text(prompt)


def ai_generate_example(word: str) -> str:
    prompt = (
        f"Provide one clear example sentence using the IELTS word '{word}'. "
        f"Return only the sentence."
    )
    return _generate_text(prompt).replace('"', "")


def ai_base_info(word: str) -> dict:
    """Used when DictionaryAPI fails. Returns {pos, cn, en_definition}."""
    prompt = (
        f"For the English word '{word}', return a JSON object with three keys: "
        f'"pos" (part of speech), "cn" (Chinese translation), '
        f'"en_definition" (a short English definition). '
        f"Return only the JSON, no markdown."
    )
    text = _generate_text(prompt)
    return _extract_json_value(text, dict)


def ai_extract_topic_tags(
    word: str, en_definition: str, allowed_tags: list[str]
) -> list[str]:
    """Pick which of allowed_tags this word relates to. Returns subset (may be [])."""
    prompt = (
        f'For the English word "{word}" (meaning: "{en_definition}"), pick which '
        f"of the following topic tags this word is most associated with. "
        f"Return a JSON array of 0–3 tags chosen ONLY from this list: "
        f"{json.dumps(allowed_tags)}. Return only the JSON array."
    )
    text = _generate_text(prompt)
    try:
        tags = _extract_json_value(text, list)
    except ValueError:
        return []
    return _dedupe_strings([t for t in tags if t in allowed_tags])


def ai_suggest_word_for_tags(tags: list[str]) -> str:
    topic = ", ".join(tags) if tags else "general academic English"
    prompt = (
        f"Suggest a single uncommon IELTS-level English word related to: {topic}. "
        f"Return only the lowercase word, no punctuation."
    )
    text = _generate_text(prompt).lower()
    match = re.search(r"[a-z][a-z-]*", text)
    return match.group(0) if match else ""


# ==========================================
# 3.5 Speaking Examiner helpers
# ==========================================
SPEAKING_TOPICS = [
    "Hometown",
    "Work or Study",
    "Hobbies and Free Time",
    "Food and Cooking",
    "Travel and Holidays",
    "Daily Routine",
    "Family and Friends",
    "Weather and Seasons",
]


def _random_topic() -> str:
    import random

    return random.choice(SPEAKING_TOPICS)


EXAMINER_SYSTEM_PROMPT = (
    "You are a friendly, professional IELTS Speaking Examiner conducting Part 1 "
    "of the IELTS Speaking test. Part 1 lasts 4–5 minutes and covers familiar, "
    "everyday topics.\n\n"
    "Instructions:\n"
    "- Start by greeting the candidate and introducing yourself briefly.\n"
    "- Ask the candidate their name, then move into the topic.\n"
    "- Ask simple, natural questions one at a time.\n"
    "- Ask occasional follow-up questions (e.g. 'Why?', 'Can you tell me more?').\n"
    "- Keep each of your messages short (1–2 sentences).\n"
    "- Do NOT correct the candidate's grammar or vocabulary during the conversation.\n"
    "- Do NOT give scores or evaluation until the session ends.\n"
    "- Cover about 5–8 questions total, then say 'Thank you. That is the end of Part 1.' "
    "when you are ready to finish.\n"
    "- Adapt naturally to the candidate's answers.\n"
    "Today's Part 1 topic is: {topic}.\n"
    "Respond in English only."
)


def _build_conversation_prompt(history: list[dict], system_prompt: str) -> str:
    """Build a single prompt string from conversation history."""
    parts = [system_prompt, ""]
    for msg in history:
        role_label = "Examiner" if msg["role"] == "examiner" else "Candidate"
        parts.append(f"{role_label}: {msg['content']}")
    # The next message will be the examiner's turn
    parts.append("Examiner:")
    return "\n".join(parts)


def _examiner_next_message(history: list[dict], topic: str) -> str:
    """Generate the examiner's next message given conversation history."""
    system_prompt = EXAMINER_SYSTEM_PROMPT.format(topic=topic)
    prompt = _build_conversation_prompt(history, system_prompt)
    return _generate_text(prompt)


def _examiner_evaluate_conversation(history: list[dict], topic: str, duration_min: float) -> dict:
    """Evaluate the full conversation and return scoring feedback."""
    transcript = "\n".join(
        f"{'Examiner' if m['role'] == 'examiner' else 'Candidate'}: {m['content']}"
        for m in history
    )
    prompt = (
        "You are an IELTS Speaking Examiner evaluating a Part 1 interview.\n\n"
        f"Topic: {topic}\n"
        f"Duration: approximately {duration_min:.1f} minutes\n\n"
        "Transcript:\n"
        f"{transcript}\n\n"
        "Evaluate the candidate's performance. Return a JSON object with these keys:\n"
        '- "overall_band": a number from 0 to 9 in 0.5 increments\n'
        '- "fluency_coherence": {{"score": number, "comment": "brief feedback in Chinese"}}\n'
        '- "lexical_resource": {{"score": number, "comment": "brief feedback in Chinese"}}\n'
        '- "grammatical_range": {{"score": number, "comment": "brief feedback in Chinese"}}\n'
        '- "strengths": [array of 2–4 Chinese strings describing strengths]\n'
        '- "improvements": [array of 2–4 Chinese strings suggesting areas to improve]\n'
        '- "notable_vocabulary": [array of English words the candidate used well, or empty array]\n'
        "Return only the JSON object, no markdown, no extra text."
    )
    text = _generate_text(prompt)
    return _extract_json_value(text, dict)
def _word_to_dict(w: DBWord) -> dict:
    return {
        "id": w.id,
        "word": w.word,
        "pos": w.pos,
        "cn": w.cn,
        "en_definition": w.en_definition,
        "example": w.example,
        "tags": _string_list_from_json(w.context),
        "image_url": _valid_image_url(w.image_path),
    }


def _definition_score(pos: str, definition: str, example: str | None) -> int:
    text = f"{definition} {example or ''}".lower()
    score = 0
    if example:
        score += 30
    if pos == "noun":
        score += 5
    obscure_terms = [
        "kruskal", "theorem", "recursive", "data structure", "connected graph",
        "finite", "vertices", "marijuana", "gallows",
    ]
    if any(term in text for term in obscure_terms):
        score -= 35
    if len(definition) > 260:
        score -= 4
    return score


def _definition_is_obscure(definition: str | None) -> bool:
    text = (definition or "").lower()
    obscure_terms = [
        "kruskal", "theorem", "recursive", "data structure", "connected graph",
        "finite", "vertices", "marijuana", "gallows",
    ]
    return any(term in text for term in obscure_terms)


def _find_dictionary_definition(dict_data: dict) -> tuple[str, str, str | None] | None:
    meanings = dict_data.get("meanings") or []
    pos_values = sorted({
        _as_text(meaning.get("partOfSpeech"), "")
        for meaning in meanings
        if isinstance(meaning, dict) and meaning.get("partOfSpeech")
    })
    candidates = []
    for meaning in meanings:
        if not isinstance(meaning, dict):
            continue
        pos = _as_text(meaning.get("partOfSpeech"), "")
        for definition in meaning.get("definitions") or []:
            if not isinstance(definition, dict):
                continue
            en_definition = _as_text(definition.get("definition"), "")
            if en_definition:
                example = _as_text(definition.get("example"), "")
                candidates.append(
                    (
                        _definition_score(pos, en_definition, example),
                        en_definition,
                        example,
                    )
                )
    if not candidates:
        return None
    _, en_definition, example = max(candidates, key=lambda c: c[0])
    return ", ".join(pos_values) or "n/a", en_definition, example


def _query_word_matching_any_tag(db: Session, selected_tags: list[str]):
    q = db.query(DBWord)
    if selected_tags:
        q = q.filter(or_(*[_context_has_tag_clause(tag) for tag in selected_tags]))
    return q.order_by(func.random()).first()


def _fallback_word_for_tags(db: Session, selected_tags: list[str]) -> DBWord:
    selected = set(selected_tags)
    data = next(
        (
            item for item in FALLBACK_WORDS
            if selected.intersection(item["tags"])
        ),
        FALLBACK_WORDS[0],
    )
    existing = db.query(DBWord).filter(DBWord.word == data["word"]).first()
    if existing:
        return existing

    word_obj = DBWord(
        word=data["word"],
        pos=data["pos"],
        cn=data["cn"],
        en_definition=data["en_definition"],
        context=json.dumps(data["tags"]),
        example=data["example"],
        image_path=None,
    )
    db.add(word_obj)
    db.commit()
    db.refresh(word_obj)
    return word_obj


def _repair_cached_word(db: Session, word_obj: DBWord) -> DBWord:
    local_info = LOCAL_WORD_INFO.get(word_obj.word)
    if local_info and (
        _cn_needs_repair(word_obj.cn)
        or _definition_is_obscure(word_obj.en_definition)
        or _definition_is_obscure(word_obj.image_path)
        or not word_obj.example
    ):
        word_obj.pos = local_info["pos"]
        word_obj.cn = local_info["cn"]
        word_obj.en_definition = local_info["en_definition"]
        word_obj.example = local_info["example"]
        word_obj.context = json.dumps(local_info["tags"])
        word_obj.image_path = None
        db.commit()
        db.refresh(word_obj)
        return word_obj

    if _cn_needs_repair(word_obj.cn):
        word_obj.cn = _fallback_translation(word_obj.word, word_obj.en_definition or "")
        db.commit()
        db.refresh(word_obj)
    return word_obj


def generate_and_save_word(db: Session, word: str) -> DBWord | None:
    """Look up + persist a word. Returns the saved DBWord, or None on full failure."""
    word = word.lower().strip()
    if not word:
        return None

    pos: str = "n/a"
    cn: str = "n/a"
    en_definition: str = "n/a"
    example: str | None = None
    base_tags: list[str] = []

    dict_data = fetch_dictionary(word)
    if dict_data:
        parsed = _find_dictionary_definition(dict_data)
        if parsed:
            pos, en_definition, example = parsed
        else:
            dict_data = None

    if not dict_data:
        try:
            base = ai_base_info(word)
        except Exception as e:
            print(f"!!! ai_base_info failed for '{word}': {e}")
            base = _offline_base_info(word)
        pos = _as_text(base.get("pos"))
        cn = _as_text(base.get("cn"))
        en_definition = _as_text(base.get("en_definition"))
        example = _as_text(base.get("example"), "") or example
        base_tags = [
            tag for tag in base.get("tags", [])
            if isinstance(tag, str) and tag in ALL_INTEREST_TAGS
        ]

    if cn == "n/a":
        cn = LOCAL_TRANSLATIONS.get(word) or fetch_free_translation(word, en_definition) or "n/a"
    if cn == "n/a" and GEMINI_API_KEY:
        try:
            cn = ai_translate(word, en_definition)
        except Exception as e:
            print(f"!!! ai_translate failed for '{word}': {e}")
            cn = "n/a"
    if _cn_needs_repair(cn):
        cn = _fallback_translation(word, en_definition)

    if not example:
        try:
            example = ai_generate_example(word)
        except Exception as e:
            print(f"!!! ai_generate_example failed for '{word}': {e}")
            example = _fallback_example(word, en_definition)

    try:
        tags = ai_extract_topic_tags(word, en_definition, ALL_INTEREST_TAGS)
    except Exception as e:
        print(f"!!! ai_extract_topic_tags failed for '{word}': {e}")
        tags = base_tags
    tags = _dedupe_strings(tags)

    new_word = DBWord(
        word=word,
        pos=pos,
        cn=cn,
        en_definition=en_definition,
        context=json.dumps(tags),
        example=example or "",
        image_path=_placeholder_image_url(),
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
if CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )


@app.get("/api/lookup/{word}")
def lookup_word(word: str, db: Session = Depends(get_db)):
    search_word = word.lower().strip()
    if not search_word:
        raise HTTPException(status_code=400, detail="Word is empty.")

    cached = db.query(DBWord).filter(DBWord.word == search_word).first()
    if cached:
        cached = _repair_cached_word(db, cached)
        return {"source": "database", "data": _word_to_dict(cached)}

    new_word = generate_and_save_word(db, search_word)
    if not new_word:
        raise HTTPException(status_code=502, detail="Failed to generate word data.")
    return {"source": "generated", "data": _word_to_dict(new_word)}


@app.post("/api/generate_image/{word}")
def generate_word_image(
    request: Request,
    word: str,
    size: str = "small",
    force: bool = False,
    db: Session = Depends(get_db),
):
    search_word = word.lower().strip()
    if not search_word:
        raise HTTPException(status_code=400, detail="Word is empty.")
    if size not in SIZE_MAP:
        raise HTTPException(status_code=400, detail=f"Invalid size '{size}'. Choose: small, medium, large.")

    quota = _check_image_quota(db, request)
    word_obj = db.query(DBWord).filter(DBWord.word == search_word).first()
    if not word_obj:
        word_obj = generate_and_save_word(db, search_word)
    if not word_obj:
        raise HTTPException(status_code=502, detail="Failed to load word data.")
    word_obj = _repair_cached_word(db, word_obj)

    if force:
        word_obj.image_path = None
        db.commit()

    result = generate_image_url(word_obj, size)
    word_obj.image_path = result["image_url"]
    word_obj.image_provider = result["provider"]
    word_obj.image_prompt = result["prompt"]
    word_obj.image_created_at = result["created_at"]
    _consume_image_quota(db, quota)
    db.commit()
    db.refresh(word_obj)

    response = {
        "provider": result["provider"],
        "data": _word_to_dict(word_obj),
    }
    if DEBUG:
        response["prompt"] = result["prompt"]
        response["created_at"] = result["created_at"]
    return response


class PreferenceRequest(BaseModel):
    selected_tags: list[str]


@app.get("/api/preferences")
def get_preferences(db: Session = Depends(get_db)):
    pref = db.query(UserPreference).first()
    if not pref:
        return {"selected_tags": DEFAULT_INTEREST_TAGS}
    return {"selected_tags": _preference_tags(pref.selected_tags)}


@app.post("/api/preferences")
def save_preferences(req: PreferenceRequest, db: Session = Depends(get_db)):
    selected_tags = _dedupe_strings(req.selected_tags)
    pref = db.query(UserPreference).first()
    if pref:
        pref.selected_tags = json.dumps(selected_tags)
    else:
        pref = UserPreference(selected_tags=json.dumps(selected_tags))
        db.add(pref)
    # Changing tags invalidates the cached "today's word".
    pref.last_daily_word_date = None
    pref.last_daily_word_id = None
    db.commit()
    return {"selected_tags": selected_tags}


@app.get("/api/daily_word")
def daily_word(db: Session = Depends(get_db)):
    today = date.today().isoformat()
    pref = db.query(UserPreference).first()
    if not pref:
        pref = UserPreference(selected_tags=json.dumps(DEFAULT_INTEREST_TAGS))
        db.add(pref)
        db.commit()
        db.refresh(pref)

    if pref.last_daily_word_date == today and pref.last_daily_word_id:
        cached = db.query(DBWord).filter(DBWord.id == pref.last_daily_word_id).first()
        if cached:
            return {"source": "cached", "data": _word_to_dict(cached)}

    selected_tags = _preference_tags(pref.selected_tags)
    word_obj: DBWord | None = None

    # 1. Try the DB first. If tags are selected, match any of them; otherwise pick
    # any random word that's already cached (avoids an AI call when we don't need one).
    word_obj = _query_word_matching_any_tag(db, selected_tags)

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

    # 4. Last resort: attempt fresh generation if API is configured, otherwise
    # seed one built-in word so a first-time local launch still has content.
    if not word_obj and GEMINI_API_KEY:
        word_obj = generate_and_save_word(db, "welcome")

    if not word_obj:
        word_obj = _fallback_word_for_tags(db, selected_tags)

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
            existing = db.query(DBWord).filter(_context_has_tag_clause(tag)).first()
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
# 5.5 Speaking Examiner endpoints
# ==========================================
class SpeakingResponse(BaseModel):
    message: str


def _session_to_dict(s: DBSpeakingSession) -> dict:
    return {
        "id": s.id,
        "topic": s.topic,
        "status": s.status,
        "created_at": s.created_at,
        "completed_at": s.completed_at,
        "evaluation": json.loads(s.evaluation) if s.evaluation else None,
    }


def _message_to_dict(m: DBSpeakingMessage) -> dict:
    return {
        "id": m.id,
        "role": m.role,
        "content": m.content,
        "created_at": m.created_at,
    }


@app.post("/api/speaking/start")
def speaking_start(db: Session = Depends(get_db)):
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY not configured.",
        )

    topic = _random_topic()
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    session = DBSpeakingSession(topic=topic, status="active", created_at=now)
    db.add(session)
    db.commit()
    db.refresh(session)

    # Generate examiner's opening message
    opening = _examiner_next_message([], topic)

    msg = DBSpeakingMessage(
        session_id=session.id,
        role="examiner",
        content=opening,
        created_at=now,
    )
    db.add(msg)
    db.commit()

    return {
        "session": _session_to_dict(session),
        "messages": [_message_to_dict(msg)],
    }


@app.post("/api/speaking/{session_id}/respond")
def speaking_respond(
    session_id: int,
    req: SpeakingResponse,
    db: Session = Depends(get_db),
):
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY not configured.",
        )

    session = db.query(DBSpeakingSession).filter(
        DBSpeakingSession.id == session_id
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    if session.status != "active":
        raise HTTPException(status_code=400, detail="Session already completed.")

    user_message = req.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Message is empty.")

    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # Save user message
    user_msg = DBSpeakingMessage(
        session_id=session_id,
        role="user",
        content=user_message,
        created_at=now,
    )
    db.add(user_msg)
    db.commit()

    # Build conversation history
    messages = (
        db.query(DBSpeakingMessage)
        .filter(DBSpeakingMessage.session_id == session_id)
        .order_by(DBSpeakingMessage.id)
        .all()
    )
    history = [{"role": m.role, "content": m.content} for m in messages]

    # Generate examiner response
    examiner_reply = _examiner_next_message(history, session.topic)

    examiner_msg = DBSpeakingMessage(
        session_id=session_id,
        role="examiner",
        content=examiner_reply,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    db.add(examiner_msg)
    db.commit()

    return {
        "session": _session_to_dict(session),
        "messages": [*[_message_to_dict(m) for m in messages], _message_to_dict(examiner_msg)],
    }


@app.post("/api/speaking/{session_id}/evaluate")
def speaking_evaluate(session_id: int, db: Session = Depends(get_db)):
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY not configured.",
        )

    session = db.query(DBSpeakingSession).filter(
        DBSpeakingSession.id == session_id
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    if session.status != "active":
        raise HTTPException(status_code=400, detail="Session already completed.")

    messages = (
        db.query(DBSpeakingMessage)
        .filter(DBSpeakingMessage.session_id == session_id)
        .order_by(DBSpeakingMessage.id)
        .all()
    )
    if len(messages) < 4:
        raise HTTPException(
            status_code=400,
            detail="Need at least 2 exchanges before evaluation.",
        )

    history = [{"role": m.role, "content": m.content} for m in messages]

    # Calculate approximate duration (assuming ~30s per exchange)
    duration_min = max(1.0, len([m for m in messages if m.role == "user"]) * 0.5)

    evaluation = _examiner_evaluate_conversation(history, session.topic, duration_min)

    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    session.status = "completed"
    session.completed_at = now
    session.evaluation = json.dumps(evaluation)
    db.commit()
    db.refresh(session)

    return {
        "session": _session_to_dict(session),
        "messages": [_message_to_dict(m) for m in messages],
    }


@app.post("/api/speaking/stt")
async def speaking_speech_to_text(audio: UploadFile = File(...)):
    """Transcribe English speech audio via Gemini."""
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY not configured.",
        )

    audio_bytes = await audio.read()
    if len(audio_bytes) < 1024:
        return {"text": "", "success": False, "error": "Audio too short"}

    _ensure_genai_configured()
    import google.generativeai as genai

    mime = audio.content_type or "audio/webm"
    print(f"[STT] Received {len(audio_bytes)} bytes, mime={mime}")

    model = genai.GenerativeModel(TEXT_MODEL_ID)

    audio_part = {"mime_type": mime, "data": audio_bytes}

    prompt = (
        "Transcribe the English speech in this audio recording. "
        "Output only the exact transcribed text, nothing else. "
        "If there is no clear English speech, output an empty string."
    )

    try:
        response = model.generate_content([prompt, audio_part])
        text = ""
        try:
            text = (response.text or "").strip()
        except Exception:
            if response.candidates:
                reason = getattr(response.candidates[0], "finish_reason", "?")
                print(f"[STT] No text — finish_reason={reason}")
        print(f"[STT] Transcription: '{text[:100]}'")
        return {"text": text, "success": True}
    except Exception as e:
        print(f"[STT] Gemini error: {e}")
        return {"text": "", "success": False, "error": str(e)}


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
