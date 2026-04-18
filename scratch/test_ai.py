import os
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

def test_extraction():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key: return

    client = genai.Client(api_key=api_key)
    
    prompt = """You are an expert Clinical Intake AI. Extract the patient data... (JSON schema)..."""
    doc = "Patient John Doe, 68yo, conditions: Diabetes. Last visit 2024-01-10."
    
    payload = [prompt, doc]

    try:
        print("Testing full extraction prompt...")
        response = client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=payload,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        print("SUCCESS!")
        data = json.loads(response.text)
        print(json.dumps(data, indent=2))
    except Exception as e:
        print(f"FAILURE: {str(e)}")

if __name__ == "__main__":
    test_extraction()
