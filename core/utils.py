import json
import os
import base64
import pandas as pd
import numpy as np
import faiss
from pypdf import PdfReader
from docx import Document
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
_client = None
if os.getenv("GEMINI_API_KEY"):
    _client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def get_ai_client():
    global _client
    if not _client and os.getenv("GEMINI_API_KEY"):
        _client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return _client

def load_json(relative_path):
    # Relative to the root folder, not utils.py
    base_dir = os.path.dirname(os.path.dirname(__file__))
    file_path = os.path.join(base_dir, relative_path)
    if not os.path.exists(file_path):
        return []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Post-process to handle base64 bytes if any
            for p in data:
                if "documents" in p:
                    for doc in p["documents"]:
                        if "raw_bytes_b64" in doc:
                            doc["raw_bytes"] = base64.b64decode(doc["raw_bytes_b64"])
            return data
    except:
        return []

def save_patients(patients):
    base_dir = os.path.dirname(os.path.dirname(__file__))
    file_path = os.path.join(base_dir, "data/patients.json")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Create a serializable copy
    export_data = []
    for p in patients:
        # Shallow copy of the patient dict
        p_copy = {k: v for k, v in p.items() if k not in ["documents", "analysis_cache"]}
        
        # Save analysis cache if serializable
        if "analysis_cache" in p:
            p_copy["analysis_cache"] = p["analysis_cache"]
            
        # Handle documents with Base64 encoding
        if "documents" in p and p["documents"]:
            export_docs = []
            for doc in p["documents"]:
                d_copy = {}
                for k, v in doc.items():
                    if k == "raw_bytes":
                        if v: d_copy["raw_bytes_b64"] = base64.b64encode(v).decode('utf-8')
                    elif isinstance(v, pd.DataFrame):
                        d_copy[k] = v.to_json() # Store DF as JSON string
                    else:
                        d_copy[k] = v
                export_docs.append(d_copy)
            p_copy["documents"] = export_docs
        else:
            p_copy["documents"] = []
            
        export_data.append(p_copy)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2)

def safe_int(val, default=0):
    """Safely convert any value to int."""
    try:
        if isinstance(val, (int, float)):
            return int(val)
        if isinstance(val, str):
            # Handle decimals in strings like "88.0"
            return int(float(val.strip()))
        return default
    except:
        return default

def get_clean_bp(bp_str):
    """Ensure BP is in systolic/diastolic format."""
    if not isinstance(bp_str, str): return "120/80"
    bp_str = bp_str.strip()
    if "/" not in bp_str: return "120/80"
    return bp_str

def read_uploaded_file(uploaded_file):
    file_name = uploaded_file.name.lower()

    try:
        if file_name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            return {"type": "structured", "name": uploaded_file.name, "content": df.to_markdown(), "preview": df.head(10)}
        if file_name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
            return {"type": "structured", "name": uploaded_file.name, "content": df.to_markdown(), "preview": df.head(10)}
        if file_name.endswith(".json"):
            data = json.load(uploaded_file)
            return {"type": "structured", "name": uploaded_file.name, "content": json.dumps(data), "preview": data}
        if file_name.endswith((".txt", ".md")):
            text = uploaded_file.read().decode("utf-8", errors="ignore")
            return {"type": "unstructured", "name": uploaded_file.name, "content": text, "preview": text[:1500]}
        if file_name.endswith(".pdf"):
            reader = PdfReader(uploaded_file)
            text = "\n".join([page.extract_text() or "" for page in reader.pages])
            return {"type": "unstructured", "name": uploaded_file.name, "content": text, "preview": text[:1500]}
        if file_name.endswith(".docx"):
            doc = Document(uploaded_file)
            text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
            return {"type": "unstructured", "name": uploaded_file.name, "content": text, "preview": text[:1500]}
        if file_name.endswith((".png", ".jpg", ".jpeg")):
            bytes_data = uploaded_file.read()
            mime = "image/png" if file_name.endswith(".png") else "image/jpeg"
            return {"type": "image", "name": uploaded_file.name, "content": f"[Image File: {uploaded_file.name}]", "preview": "Image uploaded", "raw_data": {"mime_type": mime, "data": bytes_data}}
        if file_name.endswith((".mp3", ".wav")):
            bytes_data = uploaded_file.read()
            mime = "audio/mp3" if file_name.endswith(".mp3") else "audio/wav"
            return {"type": "audio", "name": uploaded_file.name, "content": f"[Audio File: {uploaded_file.name}]", "preview": "Audio uploaded", "raw_data": {"mime_type": mime, "data": bytes_data}}
        
        return {"type": "unknown", "name": uploaded_file.name, "content": None, "preview": "Unsupported file type"}
    except Exception as e:
        return {"type": "error", "name": uploaded_file.name, "content": None, "preview": f"Error reading file: {str(e)}"}

def get_contextual_evidence(tag, patient):
    """
    Search for 1-2 line snippets in source_notes that support the given tag.
    Returns a list of evidence objects.
    """
    evidence_list = []
    source_notes = patient.get("source_notes", [])
    
    # Map high-level concepts to related keywords
    tag_keywords = {
        "Low oxygen": ["spo2", "oxygen", "saturation", "90%", "89%", "88%", "hypoxemia", "breathless", "sob"],
        "Low SpO2": ["spo2", "oxygen", "saturation", "90%", "89%", "88%", "hypoxemia"],
        "Heart rate": ["hr", "bpm", "heart rate", "pulse", "tachycardia"],
        "Glucose": ["glucose", "sugar", "diabetic", "mg/dl", "insulin"],
        "Blood pressure": ["bp", "systolic", "diastolic", "hypertension", "140/90", "pressure"],
        "Confusion": ["confused", "confusion", "disoriented", "delirium"],
        "Dizziness": ["dizzy", "dizziness", "vertigo", "lightheaded"],
        "Fatigue": ["fatigue", "tired", "lethargy", "sleepy", "somnolence", "weakness"],
        "Breathlessness": ["breathless", "sob", "short of breath", "dyspnea", "wheezing"],
        "Swelling": ["swelling", "edema", "pitting", "ankles", "legs"],
        "Appetite": ["appetite", "eating", "food", "nausea", "vomiting"],
        "Weakness": ["weak", "weakness", "frail", "strength"],
        "Fall": ["fall", "trip", "stumble", "balance", "gait"],
        "Escalate": ["urgent", "hospital", "doctor", "er", "emergency", "deterioration", "worsening"]
    }
    
    # Find matching keywords for the current tag
    keywords = next((v for k, v in tag_keywords.items() if k.lower() in tag.lower()), [])
    if not keywords:
        # Fallback to splitting the tag itself
        keywords = tag.lower().split()

    seen_contents = set()
    for note in source_notes:
        content = note.get("content", "")
        if not content or content in seen_contents: continue
        
        content_lower = content.lower()
        if any(kw in content_lower for kw in keywords):
            # Extract a snippet (approx 150 chars)
            # Find the best keyword match for the snippet center
            match_pos = -1
            found_kw = ""
            for kw in keywords:
                pos = content_lower.find(kw)
                if pos != -1:
                    match_pos = pos
                    found_kw = kw
                    break
            
            start = max(0, match_pos - 60)
            end = min(len(content), match_pos + 90)
            snippet = content[start:end].strip()
            if start > 0: snippet = "..." + snippet
            if end < len(content): snippet = snippet + "..."
            
            # Identify reason tag
            reason = "Direct Evidence"
            if any(k in content_lower for k in ["trend", "history", "previous"]): reason = "Trend Evidence"
            elif any(k in content_lower for k in ["caregiver", "wife", "son", "daughter"]): reason = "Caregiver Corroboration"

            evidence_list.append({
                "source_label": note.get("source_label", "Medical Note"),
                "file_name": note.get("source_file", "source_file"),
                "excerpt": snippet,
                "reason_tag": reason,
                "relevance_score": 0.9 # Constant for now
            })
            seen_contents.add(content)
            if len(evidence_list) >= 3: break # Max 3 snippets per finding

    return evidence_list

def assess_patient(patient, uploaded_structured, uploaded_unstructured):
    vitals = patient.get("latest_vitals", {})
    conditions = patient.get("conditions", [])
    
    risk_score = 0
    risk_factors = []
    recommended_actions = []
    evidence = []

    spo2 = safe_int(vitals.get("spo2"), 100)
    heart_rate = safe_int(vitals.get("heart_rate"), 0)
    glucose = safe_int(vitals.get("glucose"), 0)
    bp = get_clean_bp(vitals.get("blood_pressure", "120/80"))

    if spo2 < 90:
        risk_score += 3
        risk_factors.append(f"Low oxygen saturation ({spo2}%)")
        recommended_actions.append("Escalate immediately for clinical review")
        evidence.append(f"SpO2 reading of {spo2}% is critically below safe threshold (90%)")
    elif spo2 <= 92:
        risk_score += 2
        risk_factors.append(f"Borderline oxygen saturation ({spo2}%)")
        recommended_actions.append("Increase monitoring frequency")
        evidence.append(f"SpO2 reading of {spo2}% is borderline (threshold: 92%)")

    if heart_rate > 100:
        risk_score += 1
        risk_factors.append(f"Elevated heart rate ({heart_rate} bpm)")
        recommended_actions.append("Order pulse oximetry recheck")
        evidence.append(f"Heart rate of {heart_rate} bpm exceeds normal range (60-100 bpm)")

    if glucose > 200:
        risk_score += 1
        risk_factors.append(f"High glucose level ({glucose} mg/dL)")
        recommended_actions.append("Review diabetes management plan")
        evidence.append(f"Blood glucose of {glucose} mg/dL exceeds safe threshold (200 mg/dL)")

    try:
        systolic = int(bp.split("/")[0])
        diastolic = int(bp.split("/")[1])

        if systolic >= 150 or diastolic >= 95:
            risk_score += 1
            risk_factors.append(f"Elevated blood pressure ({bp})")
            recommended_actions.append("Schedule follow-up blood pressure check")
            evidence.append(f"BP reading of {bp} exceeds hypertension threshold (140/90)")
    except:
        pass

    # Aggregate all text from the synthesized clinical summary and raw source notes
    cs = patient.get("clinical_summary", {})
    summary_text = " ".join(filter(None, [
        cs.get("current_status", ""),
        cs.get("what_changed", ""),
        cs.get("observed_symptoms", ""),
        cs.get("treatment_plan", "")
    ]))
    source_notes_text = " ".join([n.get("content", "") for n in patient.get("source_notes", [])])
    combined_text = (summary_text + " " + source_notes_text).lower()
    uploaded_text_blob = " ".join(uploaded_unstructured).lower() if uploaded_unstructured else ""
    all_text = f"{combined_text} {uploaded_text_blob}"

    text_flags = {
        "confused": "New confusion reported",
        "dizziness": "Dizziness observed",
        "fatigue": "Fatigue reported",
        "breathless": "Breathlessness reported",
        "shortness of breath": "Shortness of breath reported",
        "swelling": "Swelling observed",
        "poor appetite": "Reduced appetite reported",
        "weak": "Weakness reported",
        "fall": "Fall risk signal mentioned"
    }

    for keyword, message in text_flags.items():
        if keyword in all_text:
            risk_score += 1
            risk_factors.append(message)
            evidence.append(f"Identified in care notes: '{keyword}' mentioned")

    if "Chronic Heart Failure" in conditions and ("breathless" in all_text or "shortness of breath" in all_text):
        risk_score += 2
        risk_factors.append("Heart failure patient showing respiratory deterioration")
        recommended_actions.append("Prioritize nurse or doctor escalation today")
        evidence.append("Clinical rule: CHF patient with new respiratory symptoms requires urgent escalation")

    if uploaded_structured:
        recommended_actions.append("Cross-reference uploaded structured records with current care plan")
        evidence.append("Additional structured documents provided during intake")

    if uploaded_unstructured:
        recommended_actions.append("Review uploaded clinical notes before next patient outreach")
        evidence.append("Unstructured clinical notes provided during intake")

    if risk_score >= 6:
        risk_level = "High"
        escalation = "Yes - urgent review recommended"
        priority = "P1"
    elif risk_score >= 3:
        risk_level = "Medium"
        escalation = "Maybe - monitor closely and consider follow-up"
        priority = "P2"
    else:
        risk_level = "Low"
        escalation = "No immediate escalation needed"
        priority = "P3"

    if not recommended_actions:
        recommended_actions.append("Continue routine monitoring")

    # Diversified score: use patient age + condition count to create variance
    age_factor = min(10, max(0, (patient.get('age', 50) - 50) // 5))
    cond_factor = len(conditions) * 3
    normalized_score = min(98, max(10, int(risk_score * 12 + age_factor + cond_factor + 10)))

    summary = (
        f"{patient.get('name', 'Patient')} is a {patient.get('age', '??')}-year-old {patient.get('gender', 'unknown').lower()} "
        f"with {', '.join(conditions)}. The current intake suggests a {risk_level.lower()}-risk case "
        f"based on baseline vitals, active care notes, and uploaded documents."
    )

    # ── STRUCTURED OUTPUT ──
    # Map the old flat lists to the new rich objects
    structured_risk_factors = []
    for rf in list(dict.fromkeys(risk_factors)):
        severity = "High" if rf in ["New confusion reported", "Breathlessness reported", "Heart failure patient showing respiratory deterioration"] else "Medium"
        structured_risk_factors.append({
            "title": rf,
            "severity": severity,
            "summary": f"Detected clinical indicator: {rf}",
            "evidence": get_contextual_evidence(rf, patient)
        })

    structured_actions = []
    for ra in list(dict.fromkeys(recommended_actions)):
        structured_actions.append({
            "title": ra,
            "severity": "High" if "Escalate" in ra or "Prioritize" in ra else "Medium",
            "summary": f"Recommended clinical intervention: {ra}",
            "evidence": get_contextual_evidence(ra, patient)
        })

    return {
        "summary": summary,
        "risk_level": risk_level,
        "risk_score": normalized_score,
        "priority": priority,
        "risk_factors": structured_risk_factors,
        "recommended_actions": structured_actions,
        "escalation": escalation
    }

def build_patient_context(patient, parsed_files):
    context_parts = []
    context_parts.append(f"Patient name: {patient.get('name', '??')}")
    context_parts.append(f"Age: {patient.get('age', '??')}")
    context_parts.append(f"Gender: {patient.get('gender', '??')}")
    context_parts.append(f"Conditions: {', '.join(patient.get('conditions', []))}")
    context_parts.append(f"Medications: {', '.join(patient.get('medications', []))}")
    context_parts.append(f"Last visit date: {patient.get('last_visit_date', '??')}")
    context_parts.append(f"Latest vitals: {patient.get('latest_vitals', {})}")
    
    # Synthesized clinical view
    cs = patient.get("clinical_summary", {})
    if cs.get("current_status"):
        context_parts.append(f"Current status: {cs['current_status']}")
    if cs.get("what_changed"):
        context_parts.append(f"What changed: {cs['what_changed']}")
    if cs.get("observed_symptoms"):
        context_parts.append(f"Observed symptoms: {cs['observed_symptoms']}")
    if cs.get("treatment_plan"):
        context_parts.append(f"Treatment plan: {cs['treatment_plan']}")

    # Raw source notes for full fidelity
    for note in patient.get("source_notes", []):
        context_parts.append(f"Source ({note.get('source_label', 'Note')}): {note.get('content', '')}")

    if parsed_files:
        for file in parsed_files:
            context_parts.append(f"Uploaded file: {file['name']}")
            if file["type"] == "structured":
                context_parts.append(f"Structured content preview: {str(file['preview'])[:1000]}")
            elif file["type"] == "unstructured":
                context_parts.append(f"Unstructured content preview: {str(file['preview'])[:1000]}")

    return "\n".join(context_parts)

def hash_text_to_vector(text, dim=128):
    words = text.lower().split()
    vec = np.zeros(dim, dtype=np.float32)
    for word in words:
        vec[hash(word) % dim] += 1
    # Normalize
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec

def build_faiss_index(chunks, dim=128):
    if not chunks:
        return None, []
    
    index = faiss.IndexFlatIP(dim)
    vectors = []
    for chunk in chunks:
        vectors.append(hash_text_to_vector(chunk, dim))
    
    vec_matrix = np.array(vectors).astype('float32')
    index.add(vec_matrix)
    return index, chunks

def answer_patient_question(question, patient, parsed_files):
    q = question.lower().strip()
    context = build_patient_context(patient, parsed_files)

    vitals = patient.get("latest_vitals", {})
    conditions = patient.get("conditions", [])
    medications = patient.get("medications", [])
    
    # Use basic "AI" vector search on context chunks (paragraphs) to find relevant pieces
    chunks = [c.strip() for c in context.split("\n") if len(c.strip()) > 10]
    best_matches = []
    
    if chunks:
        index, ref_chunks = build_faiss_index(chunks)
        if index is not None:
            q_vec = hash_text_to_vector(q).reshape(1, -1)
            D, I = index.search(q_vec, min(10, len(chunks)))
            if len(D) > 0 and D[0][0] > 0.05:
                # Top 10 best vector matches across all parsed docs & vitals
                best_matches = [ref_chunks[i] for i in I[0] if i < len(ref_chunks)]
                
    client = get_ai_client()
    if client and best_matches:
        try:
            prompt = f"""You are a helpful AI Care Coordinator Assistant. 
You must answer the user's question accurately using the Patient Profile and Provided Context. 
Do not hallucinate any medical data or vitals. 

Patient Profile:
- Name: {patient.get('name')}
- Age: {patient.get('age')}
- Gender: {patient.get('gender')}

Provided Context from Vector Search:
{chr(10).join(best_matches)}

User Question: {question}
Answer:"""
            
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            return response.text
        except Exception as e:
            return f"[Gemini Error: {str(e)}] Based on records, here are relevant matches:\n\n- " + "\n- ".join(best_matches)
    
    # Fallback to local exact match if no API key or no matches
    if best_matches:
        return "Based on records, here are the exact relevant matches:\n\n- " + "\n- ".join(best_matches)
    
    return "I could not find a highly confident match for that in the current patient records or documents."

def extract_patient_data_via_gemini(document_contents):
    client = get_ai_client()
    if not client:
        raise ValueError("Missing GEMINI_API_KEY. Cannot perform AI extraction.")
        
    prompt = f"""You are an expert Clinical Intake AI. Extract the patient data from the documents and unstructured text provided.
You MUST output valid JSON only, exactly matching the following schema. If a value is not found in the text, provide a sensible default (e.g. empty string or 0).

Schema:
{{
  "patient_id": "Generate a random 4 digit string like P104",
  "name": "string",
  "age": 0,
  "gender": "string",
  "conditions": ["array of strings"],
  "medications": ["array of strings"],
  "latest_vitals": {{
    "blood_pressure": "string format 120/80",
    "heart_rate": 0,
    "spo2": 0,
    "glucose": 0
  }},
  "last_visit_date": "YYYY-MM-DD",
  "clinical_summary": {{
    "current_status": "1-2 sentence synthesized overview of the patient's current state across all sources.",
    "what_changed": "What is new or different since last contact. Leave empty string if nothing significant.",
    "observed_symptoms": "Symptoms or complaints reported or observed across all sources. Leave empty string if none.",
    "treatment_plan": "Current prescribed treatment, medications, follow-up instructions from all sources."
  }},
  "source_notes": [
    {{
      "source_label": "string (e.g. Doctor Note, Nurse Note, Caregiver Message, WhatsApp Image, Lab Report, Intake Record)",
      "source_file": "string (filename or channel name, e.g. nurse_report.pdf or WhatsApp)",
      "content": "verbatim or close-paraphrase of what this specific source said"
    }}
  ]
}}
"""

    # If simple string passed, wrap in list
    if isinstance(document_contents, str):
        document_contents = [document_contents]
    
    # Translate to types.Part for multimodal support
    payload = [prompt]
    for item in document_contents:
        if isinstance(item, str):
            # Truncate large documents
            text = item[:12000] + "..." if len(item) > 12000 else item
            payload.append(text)
        elif isinstance(item, dict) and "data" in item:
            payload.append(types.Part.from_bytes(
                data=item["data"],
                mime_type=item.get("mime_type", "application/octet-stream")
            ))
        else:
            payload.append(str(item))

    import time
    
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=payload,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            data = json.loads(response.text)
            return data
        except Exception as e:
            # Internal logging
            print(f"[INTERNAL LOG] AI Extraction attempt {attempt+1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                # After all retries fail, raise a clean exception
                raise Exception("AI_SERVICE_UNAVAILABLE")
        except json.JSONDecodeError as e:
             print(f"[INTERNAL LOG] Post-AI JSON Parsing failed: {str(e)}")
             raise Exception("DATA_PARSING_ERROR")

def save_new_patient(patient_dict):
    patients = load_json("data/patients.json")
    patients.append(patient_dict)
    save_patients(patients)

def analyze_roster_question(question, patients):
    client = get_ai_client()
    if not client:
        return "⚠️ Gemini API Key missing or empty. Please configure it in the .env file."
        
    # Condense roster to save tokens and improve RAG focus
    context = ""
    for p in patients:
        context += f"Patient: {p.get('name')} ({p.get('age')} {p.get('gender')})\n"
        context += f"Conditions: {', '.join(p.get('conditions', []))}\n"
        context += f"Vitals: {p.get('latest_vitals')}\n"
        context += f"Recent Updates: {p.get('caregiver_update', '')} | {p.get('nurse_note', '')}\n\n"
        
    prompt = f"""You are the AI Care Coordinator. You are looking at a dashboard of multiple patients.
Answer the user's question accurately using ONLY this patient roster data. Be concise and actionable.

Roster Context:
{context}

Question: {question}
Answer:"""
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error querying AI API: {str(e)}"

from datetime import datetime

def archive_patient(patient_id):
    patients = load_json("data/patients.json")
    for p in patients:
        if p.get("patient_id") == patient_id:
            p["is_archived"] = True
            break
    save_patients(patients)

def add_patient_activity(patient_id, note):
    patients = load_json("data/patients.json")
    for p in patients:
        if p.get("patient_id") == patient_id:
            if "activity_log" not in p:
                p["activity_log"] = []
            p["activity_log"].insert(0, {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "note": note
            })
            break
    save_patients(patients)

def restore_patient(patient_id):
    patients = load_json("data/patients.json")
    for p in patients:
        if p.get("patient_id") == patient_id:
            p["is_archived"] = False
            break
    save_patients(patients)

def get_next_patient_id():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    file_path = os.path.join(base_dir, "data/patients.json")
    with open(file_path, "r", encoding="utf-8") as f:
        patients = json.load(f)
    max_num = 0
    for p in patients:
        pid = p.get("patient_id", "")
        if pid.startswith("P") and pid[1:].isdigit():
            max_num = max(max_num, int(pid[1:]))
    return f"P{max_num + 1:03d}"
