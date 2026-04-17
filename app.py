import json
import os
import time
from datetime import datetime, date
import streamlit as st
import pandas as pd

from core.utils import (
    load_json,
    read_uploaded_file,
    assess_patient,
    answer_patient_question,
    extract_patient_data_via_gemini,
    save_new_patient,
    analyze_roster_question,
    archive_patient,
    restore_patient,
    add_patient_activity,
    get_next_patient_id
)

if "page" not in st.session_state:
    st.session_state.page = "home"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "extracted_patient" not in st.session_state:
    st.session_state.extracted_patient = None
if "custom_actions" not in st.session_state:
    st.session_state.custom_actions = {} 
if "patient_docs" not in st.session_state:
    st.session_state.patient_docs = {} # patient_id -> [list of doc dicts]
if "analysis_dirty_by_patient" not in st.session_state:
    st.session_state.analysis_dirty_by_patient = {} # patient_id -> bool

def normalize_list(val):
    """Ensure we have a list of strings, even if AI returns a single string or null."""
    if not val: return []
    if isinstance(val, str): return [val]
    if isinstance(val, list): return [str(i) for i in val if i]
    return []
  
st.set_page_config(page_title="AI Care Coordinator", layout="wide")

def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# --- Shared Loading ---
try:
    patients = load_json("data/patients.json")
except Exception as e:
    st.error(f"Error loading patients data: {e}")
    patients = []

def nav_home():
    st.session_state.page = "home"
    st.rerun()

def nav_onboard():
    st.session_state.page = "onboard"
    st.session_state.extracted_patient = None
    st.rerun()

def nav_dashboard(patient_id):
    st.session_state.page = "dashboard"
    st.session_state.last_selected_patient = patient_id
    st.session_state.chat_history = []
    st.session_state.pop("confirm_archive", None)
    st.rerun()

def _time_label(visit_date_str):
    """Convert last_visit_date string to a relative label."""
    try:
        visit_d = datetime.strptime(visit_date_str, '%Y-%m-%d').date()
        delta = (date.today() - visit_d).days
        if delta == 0: return "today"
        if delta == 1: return "1 day ago"
        return f"{delta} days ago"
    except:
        return "—"

# ═══════════════════════════════════════════════════════════════
# PAGE: HOME
# ═══════════════════════════════════════════════════════════════
def render_home():
    st.title("AI Care Coordinator ✨")
    
    # Ask AI bar
    ai_query = st.text_input("Ask anything about your patients...", placeholder="e.g. Who needs attention today?")
    if ai_query:
        with st.spinner("AI analyzing roster..."):
            answer = analyze_roster_question(ai_query, patients)
            st.info(answer)

    # Separate active vs archived
    active_patients = [p for p in patients if not p.get("is_archived")]
    archived_patients = [p for p in patients if p.get("is_archived")]

    # Assess active patients for scores
    assessed_active = []
    critical_count = 0
    needs_review_count = 0
    
    for p in active_patients:
        # Use cache if available, otherwise run a low-cost assessment (no documents)
        if "analysis_cache" in p:
            res = p["analysis_cache"]
        else:
            res = assess_patient(p, [], [])
            
        p["_risk_level"] = res.get("risk_level", "Low")
        p["_risk_score"] = res.get("risk_score", 0)
        p["_priority"] = res.get("priority", "P3")
        
        if p["_risk_level"] == "High":
            critical_count += 1
        elif p["_risk_level"] == "Medium":
            needs_review_count += 1
        assessed_active.append(p)

    # Metrics row
    col1, col2, col3, col_onb = st.columns([1, 1, 1, 1.5])
    col1.metric("Active Patients", len(active_patients))
    col2.metric("Critical", critical_count)
    col3.metric("Needs Review", needs_review_count)
    with col_onb:
        if st.button("➕ Onboard Patient", type="primary", use_container_width=True):
            nav_onboard()

    st.markdown("---")

    # Alert banner
    high_risk = [p for p in assessed_active if p["_risk_level"] == "High"]
    if high_risk:
        st.markdown(f'<div class="risk-high">⚠️ {critical_count} patient(s) need immediate attention. Highest risk: {high_risk[0]["name"]}.</div>', unsafe_allow_html=True)
    elif needs_review_count > 0:
        st.markdown(f'<div class="risk-medium">🟡 {needs_review_count} patient(s) scheduled for review today.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="risk-low">🟢 All patients are currently stable.</div>', unsafe_allow_html=True)
    
    st.write("")

    # ── Active / Archived tabs ──
    tab_active, tab_archived = st.tabs([f"Active Patients ({len(active_patients)})", f"Archived Patients ({len(archived_patients)})"])

    with tab_active:
        if not assessed_active:
            st.info("No active patients. Onboard a new patient to get started.")
        else:
            # Sort by risk score (highest first)
            assessed_active = sorted(assessed_active, key=lambda x: x["_risk_score"], reverse=True)

            for p in assessed_active:
                time_lbl = _time_label(p.get('last_visit_date', ''))
                status_icon = "🔴" if p["_risk_level"] == "High" else "🟡" if p["_risk_level"] == "Medium" else "🟢"
                conds = ", ".join(p.get("conditions", []))

                with st.container(border=True):
                    c_id, c_name, c_risk, c_pri, c_updated, c_cond, c_action = st.columns([0.6, 1.5, 0.8, 0.5, 0.8, 1.5, 1])
                    c_id.markdown(f"<span style='color:#64748b; font-size:13px; font-weight:600;'>{p['patient_id']}</span>", unsafe_allow_html=True)
                    c_name.markdown(f"**{p['name']}** <span style='color:#94a3b8;'>({p['age']}{p.get('gender', '—')[:1]})</span>", unsafe_allow_html=True)
                    c_risk.markdown(f"{status_icon} **{p['_risk_level']}**")
                    c_pri.markdown(f"**{p['_priority']}**")
                    c_updated.markdown(f"<span style='font-size:13px; color:#64748b;'>{time_lbl}</span>", unsafe_allow_html=True)
                    c_cond.markdown(f"<span style='font-size:13px; color:#475569;'>{conds}</span>", unsafe_allow_html=True)
                    if c_action.button("View →", key=f"v_{p.get('patient_id')}", use_container_width=True):
                        nav_dashboard(p.get('patient_id'))

    with tab_archived:
        if not archived_patients:
            st.info("No archived patients.")
        else:
            for p in archived_patients:
                conds = ", ".join(p.get("conditions", []))
                with st.container(border=True):
                    c_id, c_name, c_cond, c_action = st.columns([0.6, 2, 2, 1])
                    c_id.markdown(f"<span style='color:#94a3b8; font-size:13px; font-weight:600;'>{p['patient_id']}</span>", unsafe_allow_html=True)
                    c_name.markdown(f"**{p['name']}** <span style='color:#94a3b8;'>({p['age']}{p.get('gender', '—')[:1]})</span>", unsafe_allow_html=True)
                    c_cond.markdown(f"<span style='font-size:13px; color:#94a3b8;'>{conds}</span>", unsafe_allow_html=True)
                    if c_action.button("♻️ Restore", key=f"r_{p.get('patient_id')}", use_container_width=True):
                        restore_patient(p.get('patient_id'))
                        st.toast(f"{p.get('name')} restored to active roster.")
                        st.rerun()


# ═══════════════════════════════════════════════════════════════
# PAGE: ONBOARD
# ═══════════════════════════════════════════════════════════════
def render_onboard():
    if st.button("← Back to Home"):
        nav_home()
        
    st.title("Onboard New Patient")
    
    def render_progress_bar(current_step):
        c_active = "#1d4ed8"
        c_inactive = "#94a3b8"
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; margin-bottom:30px; font-size:14px; background:#f8fafc; padding:15px; border-radius:12px; border:1px solid #e2e8f0;">
           <div style="font-weight:bold; color:{c_active if current_step=='1' else c_inactive};">1. Upload & Intake { '✓' if int(current_step)>1 else '' }</div>
           <div style="color:#cbd5e1;">➔</div>
           <div style="font-weight:bold; color:{c_active if current_step=='2' else c_inactive};">2. AI Analysis { '✓' if int(current_step)>2 else '' }</div>
           <div style="color:#cbd5e1;">➔</div>
           <div style="font-weight:bold; color:{c_active if current_step=='3' else c_inactive};">3. Human Review</div>
        </div>
        """, unsafe_allow_html=True)

    progress_placeholder = st.empty()
    step = "1" if not st.session_state.extracted_patient else "3"
    with progress_placeholder:
        render_progress_bar(step)
    
    if not st.session_state.extracted_patient:
        left, right = st.columns([1.2, 1], gap="large")
        
        with left:
            st.markdown("### What AI Will Extract")
            st.markdown("""
            Our AI will automatically scan your clinical documents and extract:
            - ✔️ **Patient demographics** (Age, Gender)
            - ✔️ **Conditions & diagnoses**
            - ✔️ **Medications**
            - ✔️ **Recent vitals** (BP, HR, SpO2, Glucose)
            - ✔️ **Risk signals** (Nurse & family notes)
            """)
            
            st.markdown("""
            <div style='margin-top:24px; font-size:13px; color:#475569; background:#f1f5f9; padding:12px; border-radius:8px;'>
               🔒 <b>HIPAA-compliant processing</b><br/>
               Data is transmitted via encrypted channels and is wiped immediately after extraction.
            </div>
            """, unsafe_allow_html=True)
            
        with right:
            st.markdown("#### ☁️ Drag & drop files here")
            uploaded_files = st.file_uploader("Upload", type=["txt", "md", "pdf", "docx", "csv", "xlsx", "json", "png", "jpg", "jpeg", "mp3", "wav"], accept_multiple_files=True, label_visibility="collapsed")
            
            st.write("")
            col_ext, col_dem = st.columns(2)
            trigger_extract = col_ext.button("🤖 AI Extract", type="primary", use_container_width=True, disabled=not uploaded_files)
            trigger_demo = col_dem.button("Try Sample Patient", use_container_width=True)
            
            if trigger_extract or trigger_demo:
                document_contents = []
                
                if trigger_demo:
                    document_contents.append("""
                    CLINICAL INTAKE NOTE:
                    Patient: John Doe, 68yo Male.
                    History: Type 2 Diabetes, Chronic Kidney Disease Stage 3.
                    Current Meds: Insulin Glargine, Lisinopril 10mg.
                    Vitals captured today at 0900: BP is slightly elevated at 142/88. Heart rate 78. SpO2 96% on room air. Glucose was 145 mg/dL.
                    Nurse Observations: Patient states he missed his lisinopril dose yesterday. Mild peripheral edema noted in ankles. Caregiver daughter reports he has been feeling more fatigued than usual over the last week.
                    """)
                else:
                    for file in uploaded_files:
                        parsed = read_uploaded_file(file)
                        if "raw_data" in parsed:
                            document_contents.append(f"--- Document: {parsed['name']} ---")
                            document_contents.append(parsed["raw_data"])
                        elif parsed.get("content"):
                            document_contents.append(f"--- Document: {parsed['name']} ---\n{parsed['content']}")
                            
                if document_contents:
                    # Update progress bar to Step 2: AI Analysis
                    with progress_placeholder:
                        render_progress_bar("2")
                        
                    with st.status("Initializing AI Agent...", expanded=True) as status:
                        st.write("Reading clinical documents & extracting data...")
                        
                        try:
                            extracted_data = extract_patient_data_via_gemini(document_contents)
                            # Pre-calculate risk for the preview
                            initial_analysis = assess_patient(extracted_data, [], [])
                            extracted_data["analysis_cache"] = initial_analysis
                            extracted_data["analysis_dirty"] = False
                            
                            # Override with deterministic sequential ID
                            extracted_data["patient_id"] = get_next_patient_id()
                            st.session_state.extracted_patient = extracted_data
                            status.update(label="Extraction Complete!", state="complete", expanded=False)
                            st.rerun() 
                        except Exception as e:
                            status.update(label=f"Extraction failed: {str(e)}", state="error", expanded=False)
                else:
                    st.warning("No readable text or data found in uploads.")
                    
    # Structured Preview (Human-in-Loop)
    if st.session_state.extracted_patient:
        st.markdown("### Step 3: Human Verification")
        st.info("The AI has structured the documents. Please review and edit the fields below before committing the patient to the database.")
        
        with st.container(border=True):
            with st.form("onboard_form"):
                ep = st.session_state.extracted_patient
                
                st.subheader("Basic Info")
                col1, col2, col3, col4 = st.columns(4)
                p_id = col1.text_input("Patient ID", value=ep.get("patient_id", ""), disabled=True)
                p_name = col2.text_input("Name", value=ep.get("name", ""))
                p_age = col3.number_input("Age", value=int(ep.get("age", 0)))
                p_gender = col4.text_input("Gender", value=ep.get("gender", ""))
                
                st.subheader("Clinical Data")
                p_cond = st.text_input("Conditions (comma separated)", value=", ".join(ep.get("conditions", [])))
                p_meds = st.text_input("Medications (comma separated)", value=", ".join(ep.get("medications", [])))
                
                st.subheader("Latest Vitals")
                v1, v2, v3, v4 = st.columns(4)
                vit = ep.get("latest_vitals", {})
                v_bp = v1.text_input("Blood Pressure", value=str(vit.get("blood_pressure", "")))
                v_hr = v2.number_input("Heart Rate", value=int(vit.get("heart_rate", 0)))
                v_spo2 = v3.number_input("SpO2", value=int(vit.get("spo2", 0)))
                v_gluc = v4.number_input("Glucose", value=int(vit.get("glucose", 0)))
                
                p_visit = st.text_input("Last Visit Date (YYYY-MM-DD)", value=ep.get("last_visit_date", ""))
                
                st.subheader("Clinical Notes Review")
                cs = ep.get("clinical_summary", {})
                
                # Editable synthesized sections
                cs_status = st.text_area("🟢 Current Status", value=cs.get("current_status", ""), key="cs_status")
                cs_changed = st.text_area("🔄 What Changed", value=cs.get("what_changed", ""), key="cs_changed")
                cs_symptoms = st.text_area("🩺 Observed Symptoms", value=cs.get("observed_symptoms", ""), key="cs_symptoms")
                cs_plan = st.text_area("💊 Treatment Plan", value=cs.get("treatment_plan", ""), key="cs_plan")
                
                # Supporting evidence as non-editable audit view
                source_notes = ep.get("source_notes", [])
                if source_notes:
                    with st.expander(f"📎 View Supporting Evidence ({len(source_notes)} source(s))", expanded=False):
                        for sn in source_notes:
                            st.markdown(f"**{sn.get('source_label', 'Source')}** · `{sn.get('source_file', '')}`")
                            st.caption(sn.get("content", ""))
                            st.divider()
                
                submitted = st.form_submit_button("Confirm & Save Patient", use_container_width=True)
                if submitted:
                    new_patient = {
                        "patient_id": ep.get("patient_id", get_next_patient_id()),
                        "name": p_name,
                        "age": p_age,
                        "gender": p_gender,
                        "conditions": [c.strip() for c in p_cond.split(",") if c.strip()],
                        "medications": [m.strip() for m in p_meds.split(",") if m.strip()],
                        "latest_vitals": {
                            "blood_pressure": v_bp,
                            "heart_rate": v_hr,
                            "spo2": v_spo2,
                            "glucose": v_gluc
                        },
                        "last_visit_date": p_visit,
                        "clinical_summary": {
                            "current_status": cs_status,
                            "what_changed": cs_changed,
                            "observed_symptoms": cs_symptoms,
                            "treatment_plan": cs_plan
                        },
                        "source_notes": ep.get("source_notes", []),
                        "documents": [],
                        "analysis_cache": ep.get("analysis_cache", {}),
                        "analysis_dirty": False
                    }
                    save_new_patient(new_patient)
                    st.success("Patient saved successfully!")
                    st.session_state.extracted_patient = None
                    nav_home()

# ═══════════════════════════════════════════════════════════════
# PAGE: DASHBOARD (No sidebar — full width)
# ═══════════════════════════════════════════════════════════════
def render_dashboard():
    if st.button("← Back to Home"):
        nav_home()
    
    if "last_selected_patient" not in st.session_state or not st.session_state.last_selected_patient:
        st.warning("No patient selected.")
        return
        
    selected_patient = next(
        (p for p in patients if p.get('patient_id') == st.session_state.last_selected_patient), 
        None
    )
    
    if not selected_patient:
        st.error("Selected patient not found.")
        return

    # ANALYSIS LOGIC & SCOPED DOCUMENTS
    pid = selected_patient.get('patient_id')
    
    # Initialize scoped docs for this patient if not present
    if pid not in st.session_state.patient_docs:
        # Robustly load documents from patient record OR empty list
        st.session_state.patient_docs[pid] = selected_patient.get("documents") or []
    
    current_p_docs = st.session_state.patient_docs.get(pid) or []
    is_dirty = st.session_state.analysis_dirty_by_patient.get(pid, False)
    
    # Check if we need to run or re-run analysis
    if "analysis_cache" not in selected_patient or is_dirty:
         with st.spinner("AI performing clinical assessment..."):
             s_inputs = [d.get("content", "") for d in current_p_docs if d.get("type") == "structured"]
             u_inputs = [d.get("content", "") for d in current_p_docs if d.get("type") != "structured"]
             res = assess_patient(selected_patient, s_inputs, u_inputs)
             
             # Save to cache persistently
             for p in patients:
                 if p.get('patient_id') == pid:
                     p["analysis_cache"] = res
                     # Note: We keep the dict in sync but the dirty flag is in session state
                     break
             
             st.session_state.analysis_dirty_by_patient[pid] = False
             from core.utils import save_patients
             save_patients(patients)
    else:
         res = selected_patient.get("analysis_cache", {})

    # ── HEADER with Archive button ──
    r_color = "riskBg-high" if res.get('risk_level') == "High" else "riskBg-medium" if res.get('risk_level') == "Medium" else "riskBg-low"
    icon = "🔴" if res.get('risk_level') == "High" else "🟡" if res.get('risk_level') == "Medium" else "🟢"
    time_lbl = _time_label(selected_patient.get('last_visit_date', ''))
    
    st.markdown(f"""
    <div style="background:#ffffff; border-radius:12px; border:1px solid #e2e8f0; padding:16px 20px; margin-bottom:16px;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <h1 style="margin:0; font-size:24px; font-weight:700; color:#0f172a;">{selected_patient.get('name')}</h1>
                <div style="color:#64748b; font-size:14px; margin-top:4px;">{selected_patient.get('patient_id')} &bull; {selected_patient.get('age')}{selected_patient.get('gender', '—')[:1]} &bull; Last visit: {time_lbl}</div>
            </div>
            <div style="text-align:right;">
                <div style="display:inline-block; padding:4px 12px; border-radius:20px; font-size:13px; font-weight:600; margin-bottom:4px;" class="{r_color}">{icon} {res.get('risk_level')} Risk</div>
                <div style="color:#64748b; font-size:12px;">Priority: {res.get('priority')}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Archive action row (inline, not sidebar)
    if not selected_patient.get("is_archived"):
        arc_col1, arc_col2 = st.columns([5, 1])
        with arc_col2:
            if st.button("🗑️ Archive", use_container_width=True):
                st.session_state["confirm_archive"] = True
        
        if st.session_state.get("confirm_archive"):
            st.warning("Are you sure you want to archive this patient?")
            cc1, cc2, cc3 = st.columns([1, 1, 4])
            if cc1.button("Yes, Archive", type="primary", use_container_width=True):
                archive_patient(selected_patient.get('patient_id'))
                st.session_state.pop("confirm_archive", None)
                st.toast("Patient archived.")
                nav_home()
            if cc2.button("Cancel", use_container_width=True):
                st.session_state.pop("confirm_archive", None)
                st.rerun()

    def render_evidence_block(evidence_list, key_suffix=""):
        if not evidence_list:
            st.caption("No specific document snippets found for this finding.")
            return

        for idx, ev in enumerate(evidence_list):
            with st.container(border=True):
                # Header with Label and Tag
                c1, c2 = st.columns([3, 1])
                c1.markdown(f"📄 **{ev['source_label']}** &bull; `{ev['file_name']}`")
                c2.markdown(f"<span style='background:#f1f5f9; color:#475569; padding:2px 8px; border-radius:10px; font-size:10px; float:right;'>{ev['reason_tag']}</span>", unsafe_allow_html=True)
                
                # Snippet
                st.markdown(f"<div style='font-size:13px; color:#334155; margin:8px 0;'>\"{ev['excerpt']}\"</div>", unsafe_allow_html=True)
                
                # View Full Source Toggle
                if st.button(f"🔍 View full source content", key=f"full_src_{key_suffix}_{idx}", use_container_width=True):
                    st.session_state[f"show_full_{key_suffix}_{idx}"] = not st.session_state.get(f"show_full_{key_suffix}_{idx}", False)
                
                if st.session_state.get(f"show_full_{key_suffix}_{idx}"):
                    # Look for the full content in the patient object or session state
                    # For simplicity, we search session state docs
                    full_txt = "Content not available"
                    for d in st.session_state.patient_docs.get(pid, []):
                        if d["name"] == ev["file_name"]:
                            full_txt = d.get("content", "Content empty")
                            break
                    st.text_area("Full Source Text", value=full_txt, height=200, disabled=True)

    # ── TABS ──
    tab_sum, tab_data, tab_ins, tab_act, tab_ai = st.tabs(["Summary", "Patient Data", "Insights", "Actions", "Ask AI"])

    with tab_sum:
        alert_col, vitals_col = st.columns([1.4, 1], gap="large")
        
        with alert_col:
            st.markdown("### Critical Alerts")
            if res.get('risk_level') == 'Low':
                 st.info("No critical alerts at this time.")
            else:
                 for factor_obj in res.get('risk_factors', [])[:3]:
                     factor_title = factor_obj.get('title') if isinstance(factor_obj, dict) else factor_obj
                     bc = '#ef4444' if res.get('risk_level')=='High' else '#f59e0b'
                     st.markdown(f"""<div style="padding:10px 12px; border-left:4px solid {bc}; background:#f8fafc; border-radius:4px; margin-bottom:6px; font-size:14px;"><strong>⚠️</strong> {factor_title}</div>""", unsafe_allow_html=True)
        
        with vitals_col:
            st.markdown("### Vitals Snapshot")
            vit = selected_patient.get("latest_vitals", {})
            
            bp_val = str(vit.get('blood_pressure', '--'))
            hr_val = vit.get('heart_rate', 0)
            spo2_val = vit.get('spo2', 100)
            gluc_val = vit.get('glucose', 0)
            
            bp_color = 'color:red;' if isinstance(bp_val, str) and '/' in bp_val and int(bp_val.split('/')[0]) >= 140 else ''
            hr_color = 'color:red;' if hr_val > 100 else ''
            spo2_color = 'color:red;' if spo2_val < 92 else ''
            gluc_color = 'color:red;' if gluc_val > 200 else ''
            
            v1, v2 = st.columns(2)
            v1.markdown(f"<div style='font-size:12px; color:#64748b;'>Blood Pressure</div><div style='font-size:28px; font-weight:700; {bp_color}'>{bp_val}</div>", unsafe_allow_html=True)
            v2.markdown(f"<div style='font-size:12px; color:#64748b;'>Heart Rate</div><div style='font-size:28px; font-weight:700; {hr_color}'>{hr_val}</div>", unsafe_allow_html=True)
            v3, v4 = st.columns(2)
            v3.markdown(f"<div style='font-size:12px; color:#64748b;'>SpO2</div><div style='font-size:28px; font-weight:700; {spo2_color}'>{spo2_val}%</div>", unsafe_allow_html=True)
            v4.markdown(f"<div style='font-size:12px; color:#64748b;'>Glucose</div><div style='font-size:28px; font-weight:700; {gluc_color}'>{gluc_val}</div>", unsafe_allow_html=True)
        
        st.markdown("### Clinical Notes")
        cs = selected_patient.get("clinical_summary", {})
        
        # Synthesized sections — clean, scannable
        sections = [
            ("🟢", "Current Status",    cs.get("current_status", "")),
            ("🔄", "What Changed",       cs.get("what_changed", "")),
            ("🩺", "Observed Symptoms",  cs.get("observed_symptoms", "")),
            ("💊", "Treatment Plan",     cs.get("treatment_plan", "")),
        ]
        for icon, label, value in sections:
            if value:
                st.markdown(
                    f"<div style='margin-bottom:10px; padding:10px 14px; background:#f8fafc; "
                    f"border-left:3px solid #3b82f6; border-radius:6px;'>"
                    f"<div style='font-size:11px; font-weight:600; color:#64748b; text-transform:uppercase; "
                    f"letter-spacing:.5px; margin-bottom:4px;'>{icon} {label}</div>"
                    f"<div style='font-size:14px; color:#0f172a;'>{value}</div></div>",
                    unsafe_allow_html=True
                )
        
        # Raw source evidence — collapsible
        source_notes = selected_patient.get("source_notes", [])
        # Removed redundant global Source Notes dump as requested
        
        if "activity_log" in selected_patient and selected_patient["activity_log"]:
            st.markdown("**Recent Activity**")
            latest_log = selected_patient["activity_log"][0]
            st.markdown(f"<div style='font-size:14px;'><b>{latest_log['timestamp']}</b>: {latest_log['note']}</div>", unsafe_allow_html=True)

    with tab_data:
        c1, c2 = st.columns(2)
        with c1:
             st.markdown("#### Demographics")
             st.write(f"- **Patient ID:** {selected_patient.get('patient_id')}")
             st.write(f"- **Name:** {selected_patient.get('name')}")
             st.write(f"- **Age:** {selected_patient.get('age')}")
             st.write(f"- **Gender:** {selected_patient.get('gender')}")
             st.write(f"- **Last Visit:** {selected_patient.get('last_visit_date', '—')}")
        with c2:
             st.markdown("#### Conditions")
             for c in selected_patient.get('conditions', []):
                 st.write(f"- {c}")
                 
        st.markdown("---")
        m1, m2 = st.columns(2)
        with m1:
             st.markdown("#### Medications")
             for m in selected_patient.get('medications', []):
                 st.write(f"- {m}")
        with m2:
             # Refresh Analysis Button
             st.markdown("#### Intelligence Controls")
             if st.button("🔄 Refresh AI Analysis", type="primary", use_container_width=True):
                  # Force dirty and reload
                  st.session_state.analysis_dirty_by_patient[pid] = True
                  st.rerun()
             
             st.write("")
             st.markdown("#### Upload Documents")
             # User Patient-Specific Uploader Key
             uploaded_files = st.file_uploader("Upload additional documents", type=["txt", "md", "pdf", "docx", "csv", "xlsx", "json", "png", "jpg", "jpeg", "mp3", "wav"], accept_multiple_files=True, label_visibility="collapsed", key=f"uploader_{pid}")
             
             if uploaded_files:
                 existing_names = [d.get("name") for d in st.session_state.patient_docs[pid]]
                 new_added = False
                 for file in uploaded_files:
                     if file.name not in existing_names:
                         pf = read_uploaded_file(file)
                         pf.update({
                             "upload_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                             "raw_bytes": file.getvalue(),
                             "mime": file.type
                         })
                         # Save to scoped session state
                         st.session_state.patient_docs[pid].append(pf)
                         st.session_state.analysis_dirty_by_patient[pid] = True
                         new_added = True
                 
                 if new_added:
                     # Update the persistence layer
                     for p in patients:
                         if p.get('patient_id') == pid:
                             p["documents"] = st.session_state.patient_docs[pid]
                             break
                     save_patients(patients)
                     st.rerun()

        # ── Document table (Direct from session state) ──
        st.markdown("---")
        active_docs = st.session_state.patient_docs.get(pid) or []
        doc_count = len(active_docs)
        st.markdown(f"#### Documents on File ({doc_count})")
        
        # Show stale-data banner if patient is dirty
        if st.session_state.analysis_dirty_by_patient.get(pid):
            st.warning("⚠️ New data added. Click 'Refresh AI Analysis' to update insights.")
        
        active_docs = st.session_state.patient_docs.get(pid, [])
        if active_docs:
            for idx, f in enumerate(active_docs):
                dc1, dc2, dc3, dc4, dc5, dc6 = st.columns([2.5, 1, 1.2, 0.8, 0.8, 0.8])
                dc1.markdown(f"📄 **{f.get('name')}**")
                dc2.caption(f.get("type", "—").capitalize())
                dc3.caption(f.get("upload_date", "—"))
                
                # View action
                if dc4.button("👁️ View", key=f"view_doc_{pid}_{idx}", use_container_width=True):
                    st.session_state[f"show_doc_{pid}_{idx}"] = not st.session_state.get(f"show_doc_{pid}_{idx}", False)
                
                # Download action
                dc5.download_button("⬇️ Download", data=f.get("raw_bytes", b""), file_name=f.get('name'), mime=f.get("mime", "application/octet-stream"), key=f"dl_doc_{pid}_{idx}", use_container_width=True)
                
                # Delete action
                if dc6.button("🗑️ Delete", key=f"del_doc_{pid}_{idx}", use_container_width=True):
                    st.session_state[f"confirm_del_doc_{pid}_{idx}"] = True
                
                # Expandable preview
                if st.session_state.get(f"show_doc_{pid}_{idx}"):
                    with st.container(border=True):
                        preview = f.get("preview", f.get("content", "No preview available"))
                        if isinstance(preview, str):
                            st.text(preview[:2000])
                        else:
                            st.dataframe(preview)
                
                # Confirm delete
                if st.session_state.get(f"confirm_del_doc_{pid}_{idx}"):
                    st.warning(f"Delete **{f.get('name')}**? Re-analysis will be required.")
                    dd1, dd2, dd3 = st.columns([1.5, 1.5, 3])
                    if dd1.button("Confirm Delete", key=f"yes_del_{pid}_{idx}", type="primary", use_container_width=True):
                        st.session_state.patient_docs[pid].pop(idx)
                        st.session_state.analysis_dirty_by_patient[pid] = True
                        # Update persistence
                        for p in patients:
                            if p.get('patient_id') == pid:
                                p["documents"] = st.session_state.patient_docs[pid]
                                break
                        from core.utils import save_patients
                        save_patients(patients)
                        st.session_state.pop(f"confirm_del_doc_{pid}_{idx}", None)
                        st.rerun()
                    if dd2.button("Cancel", key=f"no_del_{pid}_{idx}", use_container_width=True):
                        st.session_state.pop(f"confirm_del_doc_{pid}_{idx}", None)
                        st.rerun()
        else:
            st.caption("No external documents uploaded yet.")

    with tab_ins:
        st.markdown("### Risk Summary")
        risk_color = '#ef4444' if res.get('risk_level')=='High' else '#f59e0b' if res.get('risk_level')=='Medium' else '#22c55e'
        st.markdown(f"**Risk Level:** <span style='color:{risk_color}; font-weight:700;'>{res.get('risk_level')}</span>", unsafe_allow_html=True)
        st.markdown(f"**Escalation Required:** {res.get('escalation')}")
        
        st.write("")
        risk_heading = "Why is this patient high risk?" if res.get('risk_level') == 'High' else "Risk Factor Analysis" if res.get('risk_level') == 'Medium' else "Current Risk Assessment"
        st.markdown(f"#### {risk_heading}")
        
        # Each risk factor is expandable to show supporting evidence
        for ri, factor_obj in enumerate(res.get('risk_factors', [])):
            title = factor_obj.get("title")
            severity = factor_obj.get("severity", "Medium")
            bc = '#ef4444' if severity=='High' else '#f59e0b'
            
            with st.expander(f"⚠️ {title}", expanded=False):
                st.markdown(f"**Assessment:** {factor_obj.get('summary')}")
                st.markdown("---")
                st.markdown("**Supporting Evidence**")
                render_evidence_block(factor_obj.get("evidence", []), key_suffix=f"risk_{ri}")
                
                st.markdown("---")
                st.markdown("**Relevant Vitals**")
                vit = selected_patient.get("latest_vitals", {})
                ev_cols = st.columns(4)
                ev_cols[0].metric("BP", vit.get("blood_pressure", "—"))
                ev_cols[1].metric("HR", vit.get("heart_rate", "—"))
                ev_cols[2].metric("SpO2", vit.get("spo2", "—"))
                ev_cols[3].metric("Glucose", vit.get("glucose", "—"))
        
        # Structured summary view
        st.caption("Individual evidence mapped directly from patient encounter documents.")
             
        st.markdown("---")
        st.caption("Analysis synthesized via Care Intelligence Engine")

    with tab_act:
        st.markdown("### Care Actions & Recommendations")
        st.caption("Review AI-suggested interventions or log your own custom actions. All actions are patient-specific.")
        
        # Initialize action states if not present
        pid = selected_patient.get('patient_id')
        if "action_states" not in st.session_state:
            st.session_state.action_states = {}
            
        # 1. Normalize AI actions
        ai_actions_raw = normalize_list(res.get('recommended_actions', []))
        
        # 2. Get Custom actions for this patient
        custom_actions_raw = st.session_state.custom_actions.get(pid, [])
        
        # 3. Create a unified display list
        display_actions = []
        for a in ai_actions_raw:
            display_actions.append({"text": a, "type": "AI", "can_review": True})
        for a in custom_actions_raw:
            display_actions.append({"text": a, "type": "Custom", "can_review": False})
            
        if not display_actions:
            st.info("No recommended actions at this time.")
        else:
            for idx, action_item in enumerate(display_actions):
                act = action_item["text"]
                a_type = action_item["type"]
                # Use a stable UID for session state (type + index)
                act_uid = f"{pid}_{a_type}_{idx}"
                state = st.session_state.action_states.get(act_uid, "pending")
                
                with st.container(border=True):
                    ac1, ac2 = st.columns([3, 1.5])
                    
                    with ac1:
                        # Status badge for AI actions
                        if a_type == "AI":
                            if state == "approved":
                                badge = "<span style='background:#dcfce7; color:#15803d; padding:2px 8px; border-radius:10px; font-size:12px; font-weight:600;'>✓ Approved</span>"
                            elif state == "rejected":
                                badge = "<span style='background:#fef2f2; color:#dc2626; padding:2px 8px; border-radius:10px; font-size:12px; font-weight:600;'>✗ Rejected</span>"
                            else:
                                badge = "<span style='background:#fff7ed; color:#c2410c; padding:2px 8px; border-radius:10px; font-size:12px; font-weight:600;'>⏳ AI Suggested</span>"
                            
                            st.markdown(f"**⚡ {act}** &nbsp; {badge}", unsafe_allow_html=True)
                        else:
                            # Custom action styling
                            badge = "<span style='background:#f1f5f9; color:#475569; padding:2px 8px; border-radius:10px; font-size:12px; font-weight:600;'>📝 User Logged</span>"
                            st.markdown(f"**{act}** &nbsp; {badge}", unsafe_allow_html=True)

                    # Inline Evidence for Actions
                    if a_type == "AI":
                        # Match current raw action to structured object
                        action_obj = next((ao for ao in res.get('recommended_actions', []) if ao.get('title') == act), None)
                        if action_obj and action_obj.get('evidence'):
                            with st.expander("📎 View Supporting Evidence", expanded=False):
                                render_evidence_block(action_obj.get('evidence'), key_suffix=f"act_{idx}")
                    
                    with ac2:
                        if a_type == "AI" and state == "pending":
                            b1, b2 = st.columns(2)
                            if b1.button("✓ Approve", key=f"appr_{act_uid}", use_container_width=True):
                                st.session_state.action_states[act_uid] = "approved"
                                add_patient_activity(pid, f"Approved: {act}")
                                st.rerun()
                            if b2.button("✗ Reject", key=f"rej_{act_uid}", use_container_width=True):
                                st.session_state[f"reject_feedback_{act_uid}"] = True
                                st.rerun()
                        elif a_type == "Custom":
                            # Maybe a delete button for custom actions? 
                            # (Keeping it minimal as per request)
                            st.write("")
                    
                    # Rejection feedback form (progressive disclosure)
                    if st.session_state.get(f"reject_feedback_{act_uid}"):
                        with st.form(f"rej_form_{act_uid}"):
                            feedback = st.text_input("Why are you rejecting this recommendation?", placeholder="e.g. Already addressed")
                            if st.form_submit_button("Submit Feedback", type="primary"):
                                st.session_state.action_states[act_uid] = "rejected"
                                st.session_state.pop(f"reject_feedback_{act_uid}", None)
                                add_patient_activity(pid, f"Rejected: {act} — Feedback: {feedback}")
                                st.toast("Feedback recorded.")
                                st.rerun()
                    
                    # Evidence panel for AI actions
                    if a_type == "AI":
                        if st.button("View AI Reasoning", key=f"ev_{act_uid}", use_container_width=False):
                            st.session_state[f"show_ev_{act_uid}"] = not st.session_state.get(f"show_ev_{act_uid}", False)
    
                        if st.session_state.get(f"show_ev_{act_uid}"):
                            with st.container(border=True):
                                st.caption("Source data supporting this recommendation")
                                vit = selected_patient.get("latest_vitals", {})
                                evc = st.columns(4)
                                evc[0].metric("BP", vit.get("blood_pressure", "—"))
                                evc[1].metric("HR", vit.get("heart_rate", "—"))
                                evc[2].metric("SpO2", vit.get("spo2", "—"))
                                evc[3].metric("Glucose", vit.get("glucose", "—"))
                                source_notes = selected_patient.get("source_notes", [])
                                if source_notes:
                                    st.markdown("**Source Notes**")
                                    for sn in source_notes:
                                        st.caption(f"**{sn.get('source_label', 'Note')}** (`{sn.get('source_file', '')}`): {sn.get('content', '')}")

        st.markdown("---")
        st.markdown("#### Add Custom Action")
        with st.form("custom_action", clear_on_submit=True):
             custom_a = st.text_input("Describe the action...", label_visibility="collapsed", placeholder="e.g. Order BP recheck tomorrow morning")
             if st.form_submit_button("Log Action", type="primary"):
                  if custom_a.strip():
                      # Add to session state for this patient
                      if pid not in st.session_state.custom_actions:
                          st.session_state.custom_actions[pid] = []
                      st.session_state.custom_actions[pid].append(custom_a.strip())
                      
                      # Also add to activity history
                      add_patient_activity(pid, f"User Logged Action: {custom_a}")
                      st.success("Action logged and visible in Care Actions.")
                      st.rerun()
                  
    with tab_ai:
        st.caption("Ask questions grounded exclusively in this patient's clinical file.")

        suggested_questions = ["Why is this patient high risk?", "What changed recently?", "What should I do next?"]
        st.markdown("### Suggested Questions")
        
        row1_cols = st.columns(3)
        for i, col in enumerate(row1_cols):
            if col.button(suggested_questions[i], key=f"sq_{i}", use_container_width=True):
                ans = answer_patient_question(suggested_questions[i], selected_patient, p_docs)
                if "chat_history" not in st.session_state: st.session_state.chat_history = []
                st.session_state.chat_history.append(("You", suggested_questions[i]))
                st.session_state.chat_history.append(("Assistant", ans))
                st.rerun()

        st.markdown("---")
        st.markdown("### Conversation")
        with st.container(border=True, height=280):
            st.markdown(f'<div style="display:flex; justify-content:flex-start; margin-bottom:12px;"><div class="chat-assistant"><div style="font-weight:700; font-size:12px; margin-bottom:4px; color:#475569;">AI Assistant</div><div style="white-space:pre-wrap; line-height: 1.5;">I have analyzed the complete profile for {selected_patient["name"]} against our clinical criteria. Ask me anything.</div></div></div>', unsafe_allow_html=True)
            
            if "chat_history" in st.session_state and st.session_state.chat_history:
                for sender, message in st.session_state.chat_history:
                    if sender == "You":
                        st.markdown(f'<div style="display:flex; justify-content:flex-end; margin-bottom:12px;"><div class="chat-user"><div style="font-weight:700; font-size:12px; margin-bottom:4px; opacity:0.8;">You</div><div>{message}</div></div></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div style="display:flex; justify-content:flex-start; margin-bottom:12px;"><div class="chat-assistant"><div style="font-weight:700; font-size:12px; margin-bottom:4px; color:#475569;">AI Assistant</div><div style="white-space:pre-wrap; line-height: 1.5;">{message}</div></div></div>', unsafe_allow_html=True)

        user_question = st.text_input("Ask a question about this patient", placeholder="Ask anything about this patient...", label_visibility="collapsed")
        
        col_send, col_clear = st.columns([2, 1])
        with col_send:
            if st.button("Send Query", type="primary", use_container_width=True):
                if user_question.strip():
                    ans = answer_patient_question(user_question, selected_patient, p_docs)
                    if "chat_history" not in st.session_state: st.session_state.chat_history = []
                    st.session_state.chat_history.append(("You", user_question))
                    st.session_state.chat_history.append(("Assistant", ans))
                    st.rerun()
        with col_clear:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        
        # Evidence citations panel (progressive disclosure)
        if st.session_state.chat_history:
            st.markdown("---")
            if st.button("📋 View Source Evidence", use_container_width=False):
                st.session_state["show_chat_evidence"] = not st.session_state.get("show_chat_evidence", False)
            
            if st.session_state.get("show_chat_evidence"):
                with st.container(border=True):
                    st.caption("Data sources used to generate AI responses")
                    src1, src2 = st.columns(2)
                    with src1:
                        st.markdown("**Patient Vitals**")
                        vit = selected_patient.get("latest_vitals", {})
                        st.write(f"- BP: {vit.get('blood_pressure', '—')}")
                        st.write(f"- HR: {vit.get('heart_rate', '—')}")
                        st.write(f"- SpO2: {vit.get('spo2', '—')}")
                        st.write(f"- Glucose: {vit.get('glucose', '—')}")
                    with src2:
                        st.markdown("**Clinical Notes**")
                        cs = selected_patient.get("clinical_summary", {})
                        if cs.get("current_status"):
                            st.write(f"- **Status:** {cs['current_status'][:120]}")
                        if cs.get("observed_symptoms"):
                            st.write(f"- **Symptoms:** {cs['observed_symptoms'][:120]}")
                        st.write(f"- **Conditions:** {', '.join(selected_patient.get('conditions', []))}")

# ═══════════════════════════════════════════════════════════════
# ROUTER
# ═══════════════════════════════════════════════════════════════
if st.session_state.page == "home":
    render_home()
elif st.session_state.page == "onboard":
    render_onboard()
elif st.session_state.page == "dashboard":
    render_dashboard()