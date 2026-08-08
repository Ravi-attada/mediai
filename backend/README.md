---
title: MediAI Backend
emoji: 🏥
colorFrom: blue
colorTo: teal
sdk: docker
pinned: false
license: mit
---

# MediAI — AI-Powered Medical Report Analyzer Backend

Multi-Modal Deep Learning API for Disease Risk Prediction (Pneumonia & Diabetes).

## Architecture
- **NLP**: Bio_ClinicalBERT (110M params)
- **Vision**: DenseNet121 (Transfer Learning)
- **Tabular**: Clinical DNN (4-layer MLP)
- **Fusion**: Cross-Modal Attention Network

## API Endpoints
- `GET /api/health` — Health check
- `POST /api/analyze` — Full multimodal analysis
- `POST /api/analyze/text` — NLP only
- `POST /api/analyze/image` — Vision only
- `POST /api/analyze/labs` — Tabular only
