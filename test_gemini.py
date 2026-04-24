import os
import google.generativeai as genai

# --- This is the correct setup with the UPGRADED library ---
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

print("正在配置 Gemini...")
try:
    # We will explicitly tell the library to use the REST protocol to bypass proxy issues.
    genai.configure(api_key=GEMINI_API_KEY, transport='rest')
    
    # Using a standard, known-good model name.
    model = genai.GenerativeModel('models/gemini-3-flash-preview')
    print("配置成功！正在向 AI 发送请求...")
    
    response = model.generate_content("Give me a single English word about technology.")
    
    print("\n✅ AI 成功返回结果:")
    print(response.text)

except Exception as e:
    print("\n❌ 测试失败！出现了错误：")
    import traceback
    traceback.print_exc()



