import os
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

def test_gemini_3():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key: return

    try:
        client = genai.Client(api_key=api_key)
        print("Testing gemini-3-flash-preview...")
        response = client.models.generate_content(
            model="gemini-3-flash-preview", 
            contents="Say hello"
        )
        print("SUCCESS!")
        print(response.text)
    except Exception as e:
        print(f"FAILURE: {str(e)}")

if __name__ == "__main__":
    test_gemini_3()
