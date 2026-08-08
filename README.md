# 🏥 MediAI — AI-Powered Medical Report Analyzer & Disease Risk Predictor

<div align="center">

![MediAI Banner](https://img.shields.io/badge/MediAI-Multi--Modal%20AI%20System-teal?style=for-the-badge&logo=heart&logoColor=white)

[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13-orange?style=flat-square&logo=pytorch)](https://pytorch.org)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react)](https://reactjs.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.139-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-ClinicalBERT-yellow?style=flat-square&logo=huggingface)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

**MTech Research Project | Multi-Modal Deep Learning for Clinical Decision Support**

*Targeting Pneumonia Detection & Diabetes Risk Prediction — simultaneously, in one unified AI pipeline*

</div>

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Research Novelty](#-research-novelty)
- [System Architecture](#-system-architecture)
- [Deep Learning Models](#-deep-learning-models)
- [Datasets Used](#-datasets-used)
- [Project Timeline](#-project-timeline)
- [Tech Stack](#-tech-stack)
- [Installation & Running Locally](#-installation--running-locally)
- [API Endpoints](#-api-endpoints)
- [Results & Performance](#-results--performance)
- [Project Structure](#-project-structure)
- [Author](#-author)

---

## 🎯 Project Overview

**MediAI** is a full-stack, multi-modal AI system built as part of an MTech research project. It analyzes three distinct types of medical data simultaneously — **Chest X-ray images**, **Clinical Notes/Doctor reports**, and **Blood Test lab values** — and produces unified, explainable risk predictions for two diseases:

| Disease | Key Indicators Used |
|---|---|
| 🫁 **Pneumonia** | Chest X-ray + WBC, CRP, SpO₂, Temperature + Clinical text |
| 🩸 **Diabetes** | HbA1c, Blood Glucose, BMI, Cholesterol + Clinical text |

> **Key Insight:** Most existing AI medical systems handle only ONE disease from ONE data type. MediAI handles TWO diseases from THREE modalities simultaneously — a novel contribution.

---

## 🔬 Research Novelty

This project addresses **4 critical gaps** identified from a review of 25+ published papers (PubMed 2023–2025):

### Gap 1 — No Unified Multi-Disease Framework
> *"Existing papers target ONE disease only."*

**Our Solution:** A single unified pipeline predicts both **Pneumonia AND Diabetes** risk from one forward pass, sharing representational layers across diseases.

### Gap 2 — Poor Cross-Modal Explainability
> *"No paper clearly shows which modality — image, text, or labs — drove the prediction."*

**Our Solution:** A **Cross-Modal Attention Network** outputs per-modality attention weights, showing exactly how much each input (X-ray vs. clinical notes vs. blood tests) contributed to the final risk score.

### Gap 3 — Missing Modality Not Handled
> *"All existing systems require ALL inputs to be present."*

**Our Solution:** The system works with **any subset of inputs** — if a patient has no X-ray, only blood tests are used. If only clinical notes exist, they alone drive the prediction. This is critical for real-world clinical settings.

### Gap 4 — No Real-Time Clinical Deployment
> *"Research models are offline only — none are deployed as live clinical tools."*

**Our Solution:** Full-stack deployment with a **React frontend + FastAPI backend**, providing sub-second predictions through a clinical-grade web interface.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                              │
│   [Chest X-ray Image]  [Clinical Notes]  [Blood Lab Values]     │
└────────────┬──────────────────┬──────────────────┬─────────────┘
             │                  │                  │
             ▼                  ▼                  ▼
┌────────────────┐   ┌──────────────────┐   ┌──────────────────┐
│  DenseNet121   │   │  Bio_ClinicalBERT│   │   Clinical DNN   │
│  Vision Module │   │   NLP Module     │   │  Tabular Module  │
│  (PyTorch CV)  │   │  (HuggingFace)   │   │  (4-layer MLP)   │
│                │   │                  │   │                  │
│ Feature: 1024d │   │  Feature: 768d   │   │  Feature: 64d    │
└────────┬───────┘   └────────┬─────────┘   └────────┬─────────┘
         │                    │                       │
         └──────────────┬─────┴───────────────────────┘
                        │
                        ▼
         ┌──────────────────────────────┐
         │  Cross-Modal Attention       │
         │  Fusion Network (Novel)      │
         │                              │
         │  Learns which modality to    │
         │  trust most for each patient │
         └──────────────┬───────────────┘
                        │
            ┌───────────┴──────────┐
            ▼                      ▼
    ┌───────────────┐      ┌───────────────┐
    │   Pneumonia   │      │   Diabetes    │
    │   Risk Score  │      │   Risk Score  │
    │   + Attention │      │   + Attention │
    │   Attribution │      │   Attribution │
    └───────────────┘      └───────────────┘
```

---

## 🧠 Deep Learning Models

### 1. Vision Module — DenseNet121
| Property | Detail |
|---|---|
| **Architecture** | 121-layer Dense Convolutional Network |
| **Training** | Transfer Learning from ImageNet weights |
| **Fine-tuning** | Last `denseblock4` layers unfrozen |
| **Input** | 224×224 grayscale X-ray (3-channel) |
| **Output** | Pneumonia probability (sigmoid) |
| **Feature Embedding** | 1024-dimensional vector for fusion |
| **Dataset** | CheXpert (224,316) + NIH (112,120) = **336,436 X-rays** |

### 2. NLP Module — Bio_ClinicalBERT
| Property | Detail |
|---|---|
| **Model** | `emilyalsentzer/Bio_ClinicalBERT` |
| **Architecture** | BERT-Base: 12 layers, 768 hidden dims, **110M parameters** |
| **Input** | Clinical notes / discharge summaries (max 512 tokens) |
| **Output** | Dual-head: Pneumonia risk + Diabetes risk |
| **Feature Embedding** | 768-dimensional [CLS] token for fusion |
| **Dataset** | MTSamples Clinical Notes (**4,999 records**) |
| **Novel** | Single BERT backbone → two disease predictions simultaneously |

### 3. Tabular Module — Clinical DNN (Multi-task MLP)
| Property | Detail |
|---|---|
| **Architecture** | 4-layer MLP: Dense(256)→BN→Dense(128)→BN→Dense(64)→Dual Head |
| **Input** | 8 clinical lab values with **missing-value masking** |
| **Output** | Dual-head: Pneumonia risk + Diabetes risk |
| **Feature Embedding** | 64-dimensional vector for fusion |
| **Novel** | Appends binary mask vector to input — handles absent lab values natively |
| **Dataset** | PIMA Diabetes + WHO Clinical Thresholds |

### 4. Fusion Module — Cross-Modal Attention Network (Novel)
| Property | Detail |
|---|---|
| **Architecture** | Scaled dot-product attention over modality embeddings |
| **Input** | Projected embeddings from DenseNet (1024d), BERT (768d), DNN (64d) |
| **Projection** | Each modality → shared 128d latent space |
| **Attention** | Softmax-normalized weights over present modalities |
| **Output** | Weighted fused representation → Pneumonia + Diabetes predictions |
| **Missing Modality** | Attention computed only over present modalities |

---

## 📊 Datasets Used

| Dataset | Size | Type | Source |
|---|---|---|---|
| **CheXpert v1.0-small** | 224,316 X-rays | Chest X-ray images + labels | Stanford ML Group |
| **NIH Chest X-rays** | 112,120 X-rays | Chest X-ray images + 14 disease labels | NIH / Kaggle |
| **MTSamples** | 4,999 records | Clinical transcription notes | Kaggle |
| **PIMA Diabetes** | 768 records | Structured blood test data | Kaggle / UCI |
| **Combined Vision** | **336,436 X-rays** | CheXpert + NIH merged | — |

---

## 🗓️ Project Timeline

| Month | Milestone | Status |
|---|---|---|
| Month 1–2 | Literature Review (25+ PubMed papers, 2023–2025) + Dataset Acquisition | ✅ Complete |
| Month 3–4 | NLP Module — Bio_ClinicalBERT on MTSamples clinical notes | ✅ Complete |
| Month 5–6 | Vision Module — DenseNet121 on CheXpert + NIH (336K X-rays) | ✅ Complete |
| Month 7 | Multimodal Fusion — Cross-Modal Attention Network (Novel Architecture) | ✅ Complete |
| Month 8 | Explainability — Cross-Modal Attention Attribution (SHAP-style) | ✅ Complete |
| Month 9 | Web App + Deployment — FastAPI + React full-stack live system | ✅ Complete |
| Month 10 | Thesis Writing + Research Paper Submission | 🔄 In Progress |

---

## 💻 Tech Stack

### Backend
| Technology | Version | Role |
|---|---|---|
| **Python** | 3.11 | Core language |
| **FastAPI** | 0.139 | REST API framework |
| **PyTorch** | 2.13 | Deep Learning framework |
| **TorchVision** | 0.28 | DenseNet121 architecture |
| **HuggingFace Transformers** | 5.14 | Bio_ClinicalBERT model |
| **Uvicorn** | 0.51 | ASGI server |

### Frontend
| Technology | Version | Role |
|---|---|---|
| **React** | 18 | UI framework |
| **Vite** | 5 | Build tool & dev server |
| **Axios** | — | HTTP client |
| **React Hot Toast** | — | Notifications |
| **React Dropzone** | — | X-ray file upload |

---

## 🚀 Installation & Running Locally

### Prerequisites
- Python 3.11+
- Node.js 18+
- Git

### Step 1 — Clone the Repository
```bash
git clone https://github.com/Ravi-attada/mediai.git
cd mediai
```

### Step 2 — Start the Backend
```bash
cd backend
pip install -r requirements.txt
python main.py
```
Backend runs at: `http://localhost:8000`

> **Note:** First run will download Bio_ClinicalBERT weights (~450MB) from HuggingFace. This is a one-time download.

### Step 3 — Start the Frontend (new terminal)
```bash
cd frontend
npm install
npm run dev
```
Frontend runs at: `http://localhost:3000`

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/info` | Model & architecture info |
| `POST` | `/api/analyze` | **Full multimodal analysis** (main endpoint) |
| `POST` | `/api/analyze/text` | NLP module only |
| `POST` | `/api/analyze/image` | Vision module only |
| `POST` | `/api/analyze/labs` | Tabular module only |

### Example Request
```bash
curl -X POST http://localhost:8000/api/analyze \
  -F "clinical_text=Patient has fever 38.5°C, productive cough, SpO2 93%" \
  -F "blood_data={\"wbc\": 13.5, \"crp\": 68, \"blood_glucose\": 168, \"hba1c\": 7.2}"
```

### Example Response
```json
{
  "status": "success",
  "results": {
    "summary": {
      "overall_risk": "Elevated",
      "overall_score": 58.3,
      "modalities_used": ["Clinical Notes", "Blood Tests"],
      "fusion_model": "Cross-Modal Attention Network"
    },
    "pneumonia": {
      "risk_percentage": 58.3,
      "risk_level": "Moderate",
      "recommendation": "🔔 Medical consultation recommended within 24–48 hours.",
      "attention_weights": {
        "Clinical Notes (ClinicalBERT)": 14.2,
        "Blood Labs (Clinical DNN)": 17.5
      }
    },
    "diabetes": {
      "risk_percentage": 52.1,
      "risk_level": "Moderate",
      "recommendation": "🔔 Medical consultation recommended within 24–48 hours."
    }
  }
}
```

---

## 📁 Project Structure

```
mediai/
│
├── backend/                          # FastAPI + PyTorch Backend
│   ├── main.py                       # Main API entry point
│   ├── main_render.py                # Render deployment version
│   ├── requirements.txt              # Full dependencies (PyTorch)
│   ├── requirements_render.txt       # Slim dependencies (Render)
│   ├── Dockerfile                    # Docker deployment config
│   │
│   ├── models/
│   │   ├── dl/                       # Deep Learning modules (PyTorch)
│   │   │   ├── densenet_vision.py    # DenseNet121 X-ray classifier
│   │   │   ├── clinicalbert_nlp.py   # Bio_ClinicalBERT NLP module
│   │   │   ├── dnn_tabular.py        # Clinical DNN tabular module
│   │   │   └── attention_fusion.py   # Cross-Modal Attention fusion
│   │   └── fusion_model.py           # Legacy fusion (sklearn)
│   │
│   ├── train_models.py               # Train NLP + Tabular models
│   └── train_vision_combined.py      # Train Vision on CheXpert + NIH
│
├── frontend/                         # React + Vite Frontend
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Home.jsx              # Landing page
│   │   │   ├── Analyzer.jsx          # Main analysis interface
│   │   │   └── About.jsx             # Research & architecture page
│   │   ├── components/
│   │   │   ├── Navbar.jsx
│   │   │   ├── ResultCard.jsx
│   │   │   └── ExplanationChart.jsx
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── index.css                 # Glassmorphism dark theme
│   ├── vite.config.js
│   └── vercel.json
│
├── IEEE_References.md                # 22 IEEE-formatted references
├── Paper_Analyses.md                 # Literature gap analysis
└── README.md                         # This file
```

---

## 👤 Author

<div align="center">

**Ravi Attada**  
MTech Research Scholar  
📧 GitHub: [@Ravi-attada](https://github.com/Ravi-attada)

---

*This project was developed as part of an MTech thesis on Multi-Modal AI for Clinical Decision Support.*  
*All datasets used are publicly available and open-source.*

</div>
