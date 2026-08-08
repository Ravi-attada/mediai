"""
FastAPI Backend — AI-Powered Medical Report Analyzer & Disease Risk Predictor
MTech Project | Endpoints: /api/analyze | /api/analyze/text | /api/analyze/image | /api/analyze/labs
"""
import json, sys, os
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import uvicorn

from models.dl.clinicalbert_nlp  import analyze_report
from models.dl.densenet_vision   import analyze_xray
from models.dl.dnn_tabular       import analyze_blood_report
from models.dl.attention_fusion  import run_multimodal_analysis

app = FastAPI(
    title="AI-Powered Medical Report Analyzer",
    description="Multi-Modal AI for Disease Risk Prediction — Pneumonia & Diabetes",
    version="1.0.0"
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/api/health")
def health():
    return {"status": "healthy", "project": "AI-Powered Medical Report Analyzer", "version": "1.0.0"}


@app.get("/api/info")
def info():
    return {
        "title": "AI-Powered Medical Report Analyzer & Disease Risk Predictor",
        "diseases": ["Pneumonia", "Diabetes"],
        "modalities": ["Clinical Text (NLP)", "Chest X-ray (Vision)", "Blood Tests (Tabular)"],
        "novel_contributions": [
            "Cross-modal explainability with attention weights",
            "Missing modality robustness",
            "Multi-disease unified framework"
        ]
    }


@app.post("/api/analyze")
async def analyze_all(
    clinical_text: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    blood_data: Optional[str] = Form(None)
):
    nlp_result = vision_result = tabular_result = None

    if clinical_text and clinical_text.strip():
        nlp_result = analyze_report(clinical_text)

    if image and image.filename:
        image_bytes = await image.read()
        if image_bytes:
            vision_result = analyze_xray(image_bytes)

    if blood_data and blood_data.strip():
        try:
            tabular_result = analyze_blood_report(json.loads(blood_data))
        except Exception:
            pass

    if not any([nlp_result, vision_result, tabular_result]):
        raise HTTPException(400, "Please provide at least one input.")

    fusion = run_multimodal_analysis(nlp_result, vision_result, tabular_result)
    return {
        "status": "success",
        "results": fusion,
        "module_outputs": {"nlp": nlp_result, "vision": vision_result, "tabular": tabular_result},
        "input_summary": {
            "text_provided":  bool(clinical_text and clinical_text.strip()),
            "image_provided": bool(image and image.filename),
            "labs_provided":  bool(blood_data and blood_data.strip())
        }
    }


@app.post("/api/analyze/text")
async def analyze_text(payload: dict):
    text = payload.get("text", "")
    if not text.strip():
        raise HTTPException(400, "Clinical text required.")
    return {"status": "success", "module": "NLP", "results": analyze_report(text)}


@app.post("/api/analyze/image")
async def analyze_image(image: UploadFile = File(...)):
    data = await image.read()
    if not data:
        raise HTTPException(400, "Image file is empty.")
    return {"status": "success", "module": "Vision", "results": analyze_xray(data)}


@app.post("/api/analyze/labs")
async def analyze_labs(payload: dict):
    if not payload:
        raise HTTPException(400, "Blood test data required.")
    return {"status": "success", "module": "Tabular", "results": analyze_blood_report(payload)}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
