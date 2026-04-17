# Clinical Care Intelligence Prototype

A clinically-focused AI command center designed to transform scattered health data (voice, text, images, and documents) into structured, actionable insights for care teams.

![Dashboard Preview](assets/dashboard_preview.png) *(Note: Placeholder for actual screenshot)*

## 🚀 Key Features

- **Multimodal AI Intake**: Process unstructured voice recordings, handwritten notes, PDFs, and structured clinical data using Google's Gemini 2.5 Flash.
- **Auditable Clinical Insights**: Every risk factor and recommended action is backed by contextual, snippet-based evidence mapped directly to the original source documents.
- **Intelligent Dashboard**: Real-time patient monitoring with automated risk scoring (P1-P3) and critical alert highlights.
- **Contextual Search**: A dedicated "Ask AI" tab powered by Vector Search (FAISS) for precise querying of patient history and encounter notes.
- **Patient-Scoped Management**: Secure, scoped data management ensuring document and analysis integrity per patient.

## 🛠️ Technology Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **AI/LLM**: Google GenAI SDK (Gemini 2.5 Flash)
- **Vector DB**: [FAISS](https://github.com/facebookresearch/faiss)
- **Data Handling**: Pandas, Base64 (for binary persistence)
- **Environment**: Python 3.10+

## ⚙️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/care-intelligence-prototype.git
   cd care-intelligence-prototype
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment**:
   Create a `.env` file in the root directory and add your Google API Key:
   ```env
   GEMINI_API_KEY=your_actual_api_key_here
   ```

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## 🔒 Security & Privacy (HIPAA Consideration)

- Encrypted Base64 serialization is used for document storage.
- **Warning**: Ensure you do not upload real PHI (Protected Health Information) to any AI service without a Business Associate Agreement (BAA) in place.

## 📄 License

MIT License.
