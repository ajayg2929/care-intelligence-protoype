import os
import json
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

def test_high_fidelity_extraction():
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    
    prompt = """You are an expert Clinical Intake AI. Extract the patient data from the documents and unstructured text provided.
You MUST output valid JSON only, exactly matching the following schema.

Schema:
{
  "patient_id": "P104",
  "name": "string",
  "age": 0,
  "gender": "string",
  "conditions": ["array of strings"],
  "medications": ["array of strings"],
  "latest_vitals": {
    "blood_pressure": "120/80",
    "heart_rate": 0,
    "spo2": 0,
    "glucose": 0
  },
  "last_visit_date": "YYYY-MM-DD",
  "clinical_summary": {
    "current_status": "string",
    "what_changed": "string",
    "observed_symptoms": "string",
    "treatment_plan": "string"
  },
  "source_notes": [
    {
      "source_label": "string",
      "source_file": "string",
      "content": "string"
    }
  ]
}
"""
    doc = """
    CLINICAL INTAKE NOTE:
    Patient: John Doe, 68yo Male.
    History: Type 2 Diabetes, Chronic Kidney Disease Stage 3.
    Current Meds: Insulin Glargine, Lisinopril 10mg.
    Vitals: BP 142/88. HR 78. SpO2 96%. Glucose 145.
    """
    
    payload = [prompt, doc]

    print("Running high-fidelity extraction test...")
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=payload,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        print("SUCCESS!")
        print(response.text)
    except Exception as e:
        print(f"FAILURE: {str(e)}")

if __name__ == "__main__":
    test_high_fidelity_extraction()
