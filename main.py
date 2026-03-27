import os
import json
import re
import traceback
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String
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
    pos = Column(String); cn = Column(String); context = Column(String)
    example = Column(String); video_url = Column(String); image_url = Column(String)

Base.metadata.create_all(bind=engine)

# ==========================================
# 2. Gemini 2026 旗舰多模态配置
# ==========================================
GEMINI_API_KEY = "AIzaSyBlUQ9mH8idzvEK0PGXOYvWmHb_r9AQGAg" 
genai.configure(api_key=GEMINI_API_KEY.strip(), transport='rest')

TEXT_MODEL_ID = 'models/gemini-3-flash-preview'
IMAGE_MODEL_ID = 'models/nano-banana-pro-preview' 
VIDEO_MODEL_ID = 'models/veo'

def safe_get_media_url(response, default_url):
    """安全提取生成媒体的 URL"""
    try:
        # 尝试从候选结果中提取 URI 或文本链接
        part = response.candidates[0].content.parts[0]
        # 优先寻找文件 URI，如果没有则尝试 text
        url = getattr(part, 'file_data', None)
        if url: return url.file_uri
        return getattr(part, 'text', default_url)
    except:
        return default_url

# ==========================================
# 3. FastAPI 核心逻辑
# ==========================================
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/api/lookup/{word}")
def lookup_word(word: str):
    db = SessionLocal()
    search_word = word.lower().strip()
    try:
        # 1. 查库
        db_word = db.query(DBWord).filter(DBWord.word == search_word).first()
        if db_word: return {"source": "database", "data": db_word}
        
        # 2. 文本语义分析
        model = genai.GenerativeModel(TEXT_MODEL_ID)
        prompt = f"Analyze '{search_word}' and return JSON with: word, pos, cn, context, example, visual_prompt(EN)."
        response = model.generate_content(prompt)
        
        # 格式化解析 JSON
        res_text = response.text.strip()
        json_match = re.search(r'\{.*\}', res_text, re.DOTALL)
        if not json_match: raise ValueError(f"AI没有返回JSON: {res_text}")
        ai_data = json.loads(json_match.group(0))
        
        # 3. 多模态生成 (增加独立异常保护)
        v_prompt = ai_data.get("visual_prompt", f"Educational visual for {search_word}")
        
        # 生成图片
        try:
            img_res = genai.GenerativeModel(IMAGE_MODEL_ID).generate_content(f"Realistic image: {v_prompt}")
            image_url = safe_get_media_url(img_res, "https://picsum.photos/800/600")
        except: image_url = "https://picsum.photos/800/600"

        # 生成视频
        try:
            vid_res = genai.GenerativeModel(VIDEO_MODEL_ID).generate_content(f"Cinematic video: {v_prompt}")
            video_url = safe_get_media_url(vid_res, "https://www.w3schools.com/html/mov_bbb.mp4")
        except: video_url = "https://www.w3schools.com/html/mov_bbb.mp4"

        # 4. 存库
        new_word = DBWord(
            word=search_word,
            pos=ai_data.get("pos", "n."),
            cn=ai_data.get("cn", "含义加载中"),
            context=ai_data.get("context", "通用"),
            example=ai_data.get("example", ""),
            image_url=image_url,
            video_url=video_url
        )
        db.add(new_word)
        db.commit()
        db.refresh(new_word)
        return {"source": "ai_generated_multimodal", "data": new_word}

    except Exception as e:
        db.rollback()
        print("\n❌ [ERROR DETAILS]:")
        traceback.print_exc() # 在终端打印最详细的报错行数
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

class ChatRequest(BaseModel): user_message: str
@app.post("/api/chat")
def ai_chat(request: ChatRequest):
    try:
        model = genai.GenerativeModel(TEXT_MODEL_ID)
        res = model.generate_content(request.user_message)
        return {"reply": res.text}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))