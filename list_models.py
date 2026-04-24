import os
import google.generativeai as genai
import traceback

# Use the exact same setup that we've been testing
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

print("正在连接 Google AI 以获取可用模型列表...")
try:
    # We will use the same REST transport setting
    genai.configure(api_key=GEMINI_API_KEY, transport='rest')
    
    print("-" * 30)
    print("以下是您当前可用的模型：")
    
    found_models = False
    for m in genai.list_models():
        # We only care about models that can generate text
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)
            found_models = True
            
    if not found_models:
        print("没有找到支持 'generateContent' 的模型。")

    print("-" * 30)

except Exception as e:
    print("\n❌ 获取模型列表失败！出现了错误：")
    traceback.print_exc()
