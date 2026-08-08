# 🏥 AI-Powered Medical Report Analyzer & Disease Risk Predictor

**MTech Project** | Multi-Modal AI System for Automated Clinical Report Analysis and Early Disease Risk Prediction using NLP and Deep Learning

---

## 🎯 Diseases Covered
- 🫁 **Pneumonia** — Chest X-ray + Blood tests + Clinical notes
- 🩸 **Diabetes** — Blood glucose, HbA1c, BMI + Clinical notes

## 🔬 Novel Contributions
1. **Cross-modal explainability** — SHAP showing which modality drove prediction
2. **Missing modality robustness** — Works with partial inputs
3. **Multi-disease unified framework** — One model for both diseases

---

## 🚀 How to Run

### Backend (FastAPI)
```bash
cd backend
pip install -r requirements.txt
python main.py
```
Backend runs at: http://localhost:8000
API Docs: http://localhost:8000/docs

### Frontend (React)
```bash
cd frontend
npm install
npm run dev
```
Frontend runs at: http://localhost:3000

---

## 📁 Project Structure
```
medical-ai-analyzer/
├── backend/
│   ├── main.py                  ← FastAPI app
│   ├── models/
│   │   ├── nlp_model.py         ← NLP text analysis (Month 3-4)
│   │   ├── vision_model.py      ← X-ray analysis (Month 5-6)
│   │   ├── tabular_model.py     ← Blood test analysis (Month 3-4)
│   │   └── fusion_model.py      ← Multimodal fusion (Month 7)
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── components/          ← Reusable UI components
│       ├── pages/               ← Home, Analyzer, About
│       └── App.jsx
└── README.md
```

## 📊 Datasets
- **MIMIC-CXR** — PhysioNet (free with registration)
- **CheXpert** — Stanford ML Group
- **MIMIC-III** — PhysioNet
- **PIMA Diabetes** — Kaggle/UCI
