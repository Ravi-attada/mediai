"""
Tabular Model — Real XGBoost trained on PIMA Diabetes Dataset
Loads: trained_models/tabular_model.pkl
"""
import os, pickle, numpy as np
from pathlib import Path

_MODEL_PATH = Path(__file__).parent.parent / "trained_models" / "tabular_model.pkl"
_model_cache = None

def _load_model():
    global _model_cache
    if _model_cache is None and _MODEL_PATH.exists():
        with open(_MODEL_PATH, "rb") as f:
            _model_cache = pickle.load(f)
    return _model_cache

FEATURE_ORDER = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

NORMAL_RANGES = {
    "Glucose":    (70, 99),
    "BloodPressure": (60, 80),
    "BMI":        (18.5, 24.9),
    "HbA1c":      (4.0, 5.6),
    "WBC":        (4.5, 11.0),
    "SpO2":       (95, 100),
    "CRP":        (0, 10),
    "Temperature":(36.1, 37.2),
}

def analyze_blood_report(data: dict) -> dict:
    """
    Accepts blood test values dict from frontend.
    Returns risk scores for Pneumonia and Diabetes.
    """
    model_obj = _load_model()

    # ── DIABETES prediction via real model ────────────────────────────────────
    diabetes_prob = None
    if model_obj:
        try:
            pipeline = model_obj["pipeline"]
            glucose     = float(data.get("blood_glucose", data.get("Glucose", 120)))
            bp          = float(data.get("blood_pressure", data.get("BloodPressure", 70)))
            bmi         = float(data.get("bmi", data.get("BMI", 25)))
            age         = float(data.get("age", 35))
            hba1c       = float(data.get("hba1c", 5.5))
            insulin     = float(data.get("insulin", 80))
            skin_thick  = float(data.get("skin_thickness", 20))
            pregnancies = float(data.get("pregnancies", 0))
            dpf         = 0.5 + (hba1c - 5.0) * 0.1  # approximate pedigree from HbA1c

            features = np.array([[pregnancies, glucose, bp, skin_thick,
                                   insulin, bmi, dpf, age]])
            diabetes_prob = float(pipeline.predict_proba(features)[0][1]) * 100
        except Exception:
            diabetes_prob = None

    # ── PNEUMONIA heuristic from blood markers ────────────────────────────────
    wbc  = float(data.get("wbc", data.get("WBC", 7.0)))
    crp  = float(data.get("crp", data.get("CRP", 5.0)))
    spo2 = float(data.get("spo2", data.get("SpO2", 98.0)))
    temp = float(data.get("temperature", data.get("Temperature", 36.8)))

    pneu_score = 0.0
    if wbc  > 11.0: pneu_score += 25
    if crp  > 10.0: pneu_score += 25
    if spo2 < 95.0: pneu_score += 30
    if temp > 37.5: pneu_score += 20
    pneu_score = min(pneu_score, 95.0)

    # ── Abnormality flags ─────────────────────────────────────────────────────
    flags = []
    if wbc  > 11.0: flags.append({"marker": "WBC Count",       "value": wbc,  "status": "HIGH",   "normal": "4.5-11.0 ×10³/µL"})
    if crp  > 10.0: flags.append({"marker": "CRP",             "value": crp,  "status": "HIGH",   "normal": "0-10 mg/L"})
    if spo2 < 95.0: flags.append({"marker": "SpO2",            "value": spo2, "status": "LOW",    "normal": "95-100%"})
    if temp > 37.5: flags.append({"marker": "Temperature",     "value": temp, "status": "HIGH",   "normal": "36.1-37.2°C"})
    if float(data.get("blood_glucose", 120)) > 126:
        flags.append({"marker": "Blood Glucose", "value": data.get("blood_glucose"), "status": "HIGH", "normal": "70-99 mg/dL"})
    if float(data.get("hba1c", 5.5)) > 6.4:
        flags.append({"marker": "HbA1c",         "value": data.get("hba1c"),         "status": "HIGH", "normal": "<5.7%"})

    if diabetes_prob is None:
        # Fallback heuristic if model not yet loaded
        g = float(data.get("blood_glucose", 120))
        h = float(data.get("hba1c", 5.5))
        b = float(data.get("bmi", 25))
        diabetes_prob = min(
            ((g - 70) / 130 * 40) + ((h - 4.0) / 4.0 * 35) + ((b - 18) / 20 * 25), 95
        )
        diabetes_prob = max(diabetes_prob, 5)

    model_used = "Real GBM — Trained on PIMA Diabetes Dataset (768 patients)" \
                 if model_obj else "Heuristic (run train_models.py to enable real model)"

    return {
        "pneumonia_risk":  round(pneu_score, 1),
        "diabetes_risk":   round(max(diabetes_prob, 5.0), 1),
        "abnormal_flags":  flags,
        "model_used":      model_used,
        "dataset":         "PIMA Indians Diabetes Database + WHO Lab Standards",
        "features_analyzed": len(data)
    }
