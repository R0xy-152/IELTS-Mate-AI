import os
import json
import re
import requests
from datetime import date
import traceback
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, func
from sqlalchemy.orm import declarative_base, sessionmaker

# ==========================================
# 0. 环境与代理
# ==========================================
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"

# ==========================================
# 1. 数据库配置
# ==========================================
SQLALCHEMY_DATABASE_URL = "sqlite:///./ielts_mate.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class DBWord(Base):
    __tablename__ = "words"
    id = Column(Integer, primary_key=True, index=True)
    word = Column(String, unique=True, index=True)
    pos = Column(String)
    cn = Column(String)
    en_definition = Column(String)
    context = Column(String)
    example = Column(String)
    video_url = Column(String)
    image_url = Column(String)

class UserPreference(Base):
    __tablename__ = "user_preferences"
    id = Column(Integer, primary_key=True, index=True)
    selected_tags = Column(String, default="[]")
    last_daily_word_date = Column(String)
    last_daily_word_id = Column(Integer)

Base.metadata.create_all(bind=engine)

# ==========================================
# 2. Gemini & AI 配置
# ==========================================
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY, transport='rest')

TEXT_MODEL_ID = 'models/gemini-3-flash-preview'
IMAGE_MODEL_ID = 'models/nano-banana-pro-preview'
VIDEO_MODEL_ID = 'models/nano-banana-pro-preview' # Placeholder

def safe_get_media_url(response, default_url):
    try:
        part = response.candidates[0].content.parts[0]
        url = getattr(part, 'file_data', None)
import traceback
import time # 新增
from google.api_core.exceptions import TooManyRequests # 新增
import google.generativeai as genai

# ... (省略中间代码) ...

# ==========================================
# 3. FastAPI 核心应用
# ==========================================
app = FastAPI()

# ------------------------------------------
# (新增) 带有重试逻辑的AI调用辅助函数
# ------------------------------------------
def call_gemini_with_retry(model, prompt):
    try:
        # 第一次尝试
        return model.generate_content(prompt)
    except TooManyRequests as e:
        print("!!! [AI QUOTA] 429 Error: Resource has been exhausted. Waiting 60 seconds before retrying...")
        time.sleep(60)
        # 第二次尝试
        print("--> [AI RETRY] Retrying the request after 60 seconds...")
        return model.generate_content(prompt)

# ------------------------------------------
# 辅助函数: 生成并保存单词 (混合模式 V2)
# ------------------------------------------
def generate_and_save_word(db, word_to_generate):
    print(f"--> [Hybrid V2] Attempting to generate and save word: '{word_to_generate}'")
    
    pos, cn, en_definition, context, example = "n/a", "n/a", "n/a", "n/a", None
    
    try:
        # 步骤 1: 调用 DictionaryAPI
        dict_api_url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word_to_generate}"
        response = requests.get(dict_api_url)
        
        if response.status_code == 200:
            data = response.json()[0]
            pos = ", ".join(sorted(list(set(m['partOfSpeech'] for m in data['meanings']))))
            en_definition = data['meanings'][0]['definitions'][0]['definition']
            example = data['meanings'][0]['definitions'][0].get('example')
            context = pos 
            print(f"--> [DictAPI] OK: POS, EN_DEF, Example for '{word_to_generate}'.")

            # 步骤 2a: 调用 AI 进行翻译
            try:
                model_cn = genai.GenerativeModel(TEXT_MODEL_ID)
                prompt_cn = f"Translate the word '{word_to_generate}' to Chinese. English definition for context: '{en_definition}'."
                response_cn = model_cn.generate_content(prompt_cn)
                cn = response_cn.text.strip()
            except Exception as e:
                print(f"!!! [AI WARNING] Chinese translation failed: {e}")
                cn = "翻译失败"
        else:
            # 步骤 1/2b: 降级 - 如果 DictionaryAPI 失败，则用 AI 获取基础信息
            print(f"!!! [DictAPI WARNING] Failed. Downgrading to pure AI for base info.")
            try:
                model_base = genai.GenerativeModel(TEXT_MODEL_ID)
                prompt_base = f"Provide part of speech (pos), Chinese translation (cn), and a simple English definition (en_definition) for the word: '{word_to_generate}'. Return JSON."
                response_base = model_base.generate_content(prompt_base)
                base_data = json.loads(re.search(r'\{.*\}', response_base.text, re.DOTALL).group(0))
                pos, cn, en_definition = base_data.get('pos', 'n/a'), base_data.get('cn', 'n/a'), base_data.get('en_definition', 'n/a')
                context = pos
            except Exception as e:
                print(f"!!! [AI FATAL] Base info generation failed. Aborting for this word. {e}")
                raise e

        # 步骤 3: 如果没有例句，则调用 AI 生成
        if not example:
            print("--> No example from DictAPI. Calling AI to generate one.")
            try:
                model_ex = genai.GenerativeModel(TEXT_MODEL_ID)
                prompt_ex = f"Provide a simple and clear example sentence for the IELTS word: '{word_to_generate}'."
                response_ex = model_ex.generate_content(prompt_ex)
                example = response_ex.text.strip().replace('"', '')
            except Exception as e:
                print(f"!!! [AI WARNING] Example generation failed: {e}")
                example = "AI example generation failed."

        # 步骤 4: 图片生成 (使用统一的 prompt)
        image_url = "" # 设置为空字符串，以便前端可以激活备用方案
            print(f"!!! [AI WARNING] Image generation failed: {e}")

        # 步骤 5: 数据保存
        new_word = DBWord(
            word=word_to_generate,
            pos=pos, cn=cn, en_definition=en_definition,
            context=context, example=example,
            image_url=image_url,
            video_url="https://www.w3schools.com/html/mov_bbb.mp4"
        )
        db.add(new_word)
        db.commit()
        db.refresh(new_word)
        print(f"--> [DB] OK: Saved '{word_to_generate}'.")
        return new_word
        
    except Exception as e:
        print(f"!!! [FATAL ERROR] An error occurred inside generate_and_save_word for '{word_to_generate}':")
        traceback.print_exc()
        return None

# ------------------------------------------
# API 路由
# ------------------------------------------
@app.get("/api/lookup/{word}")
def lookup_word(word: str):
    db = SessionLocal()
    try:
        search_word = word.lower().strip()
        db_word = db.query(DBWord).filter(DBWord.word == search_word).first()
        if db_word: return {"source": "database", "data": db_word}
        
        ai_generated_word = generate_and_save_word(db, search_word)
        if ai_generated_word:
            return {"source": "ai_generated_multimodal", "data": ai_generated_word}
        else:
            raise HTTPException(status_code=500, detail="Failed to generate word data.")
    except Exception as e:
        db.rollback()
        print(f"!!! [API ERROR in /lookup/{word}]")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error in lookup_word: {e}")
    finally:
        db.close()

@app.get("/api/daily_word")
def get_daily_word():
    print("\n--- Request received for /api/daily_word ---")
    db = SessionLocal()
    try:
        today_str = date.today().isoformat()
        pref = db.query(UserPreference).first()

        if not pref:
            print("Step 1: No preferences found. Creating new entry.")
            pref = UserPreference(selected_tags="[]")
            db.add(pref)
            db.commit()
            db.refresh(pref)
        
        print("Step 2: Checking cache for today...")
        if pref.last_daily_word_date == today_str and pref.last_daily_word_id:
            cached_word = db.query(DBWord).filter(DBWord.id == pref.last_daily_word_id).first()
            if cached_word:
                print("--> Cache hit! Returning cached word.")
                return {"source": "cached_daily", "data": cached_word}

        word_to_return = None
        selected_tags = json.loads(pref.selected_tags)
        print(f"Step 3: No cache. Searching DB with tags: {selected_tags}")
        
        if selected_tags:
            query = db.query(DBWord)
            for tag in selected_tags:
                query = query.filter(DBWord.context.like(f"%{tag}%"))
            word_to_return = query.order_by(func.random()).first()

        if not word_to_return:
            print("Step 4: No DB result. Attempting AI generation for a suggestion...")
            try:
                model = genai.GenerativeModel(TEXT_MODEL_ID)
                tags_str = ", ".join(selected_tags) if selected_tags else "general knowledge"
                prompt = f"Give me a single, uncommon IELTS word related to these topics: {tags_str}. Only return the word itself, nothing else."
                response = model.generate_content(prompt)
                new_word_str = response.text.strip().lower().replace('.', '')
                
                print(f"--> AI suggested word: '{new_word_str}'")
                existing_word = db.query(DBWord).filter(DBWord.word == new_word_str).first()
                if existing_word:
                    print("--> Suggested word already exists in DB. Using existing entry.")
                    word_to_return = existing_word
                else:
                    print("--> Suggested word is new. Generating full content...")
                    word_to_return = generate_and_save_word(db, new_word_str)
            except Exception as ai_suggestion_error:
                print(f"!!! [AI SUGGESTION ERROR] The first AI call failed: {ai_suggestion_error}")
                word_to_return = None
        
        print("Step 5: Checking safety net...")
        if not word_to_return:
            print("--> AI generation failed. Safety net triggered.")
            
            if selected_tags:
                print(f"--> Safety net: Searching DB for pre-cached word with tags: {selected_tags}")
                query = db.query(DBWord)
                for tag in selected_tags:
                    query = query.filter(DBWord.context.like(f"%{tag}%"))
                word_to_return = query.order_by(func.random()).first()

            if not word_to_return:
                print("--> Safety net: No tag-specific word found. Trying any random word from DB.")
                word_to_return = db.query(DBWord).order_by(func.random()).first()

            if not word_to_return:
                print("--> Safety net: DB is empty. Generating 'welcome' as a last resort.")
                word_to_return = generate_and_save_word(db, "welcome")
        
        print(f"Step 6: Final word is '{word_to_return.word}'. Updating cache.")
        pref.last_daily_word_date = today_str
        pref.last_daily_word_id = word_to_return.id
        db.commit()
        
        print("--- Request successful. Returning word. ---")
        return {"source": "new_daily", "data": word_to_return}

    except Exception as e:
        db.rollback()
        print("!!! [FATAL ERROR in /api/daily_word] An exception was caught:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"A server error occurred in get_daily_word: {e}")
    finally:
        db.close()

class PreferenceRequest(BaseModel):
    selected_tags: list[str]

@app.post("/api/preferences")
def save_preferences(req: PreferenceRequest):
    print("\n--- Request received for /api/preferences ---")
    db = SessionLocal()
    try:
        pref = db.query(UserPreference).first()
        if pref:
            print("--> Found existing preferences. Updating tags.")
            pref.selected_tags = json.dumps(req.selected_tags)
        else:
            print("--> No preferences found. Creating new entry.")
            pref = UserPreference(selected_tags=json.dumps(req.selected_tags))
            db.add(pref)
        
        print("--> Clearing daily word cache due to preference change.")
        pref.last_daily_word_date = None
        pref.last_daily_word_id = None
        
        db.commit()
        print("--- Preferences saved and cache cleared successfully. ---")
        return {"selected_tags": req.selected_tags}
    except Exception as e:
        db.rollback()
        print(f"!!! [FATAL ERROR in /api/preferences]")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()
        
# ------------------------------------------
# 数据库预缓存功能
# ------------------------------------------
ALL_INTEREST_TAGS = [
    "Architecture", "Esports", "Anime", "Music", "Sports", "Board Games",
    "Programming", "Baking", "Business", "Technology", "Travel", "Food & Dining",
    "Nature", "Art", "History", "Health", "Society", "Finance", "Law",
    "Education", "Media", "Environment"
]

@app.get("/api/pre-cache-tags")
def pre_cache_tags():
    print("\n--- Request received for /api/pre-cache-tags ---")
    db = SessionLocal()
    generated_count = 0
    failed_tags = []
    try:
        for tag in ALL_INTEREST_TAGS:
            try:
                print(f"--> Checking cache for tag: '{tag}'")
                existing_word = db.query(DBWord).filter(DBWord.context.like(f"%{tag}%")).first()
                
                if not existing_word:
                    print(f"--> No word found for '{tag}'. Generating a new one...")
                    model = genai.GenerativeModel(TEXT_MODEL_ID)
                    prompt = f"Give me a single, representative IELTS word related to the topic: {tag}. Only return the word itself, nothing else."
                    response = model.generate_content(prompt)
                    new_word_str = response.text.strip().lower().replace('.', '')

                    if not db.query(DBWord).filter(DBWord.word == new_word_str).first():
                        saved_word = generate_and_save_word(db, new_word_str)
                        if saved_word:
                            generated_count += 1
                            print(f"--> Successfully generated and saved '{saved_word.word}' for '{tag}'.")
                        else:
                            raise Exception(f"generate_and_save_word returned None for {new_word_str}")
                    else:
                        print(f"--> AI suggested an existing word for '{tag}'. Skipping.")
                else:
                    print(f"--> Cache hit for '{tag}'. Skipping.")
            except Exception as tag_error:
                print(f"!!! [ERROR] Failed to process tag '{tag}': {tag_error}")
                failed_tags.append(tag)
                continue

        summary_message = f"Pre-cache process complete. Generated {generated_count} new words. Failed tags: {failed_tags}"
        print(f"--- {summary_message} ---")
        return {"message": summary_message, "generated_count": generated_count, "failed_tags": failed_tags}
        
    except Exception as e:
        print(f"!!! [FATAL ERROR in /api/pre-cache-tags]")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

# ------------------------------------------
# 静态文件挂载 (必须在所有路由之后)
# ------------------------------------------
app.mount("/", StaticFiles(directory=".", html=True), name="static")
