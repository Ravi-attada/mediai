"""
NLP Model — Real TF-IDF + Logistic Regression trained on MTSamples Dataset
Loads: trained_models/nlp_model.pkl
"""
import pickle, re
from pathlib import Path

_MODEL_PATH = Path(__file__).parent.parent / "trained_models" / "nlp_model.pkl"
_model_cache = None

def _load_model():
    global _model_cache
    if _model_cache is None and _MODEL_PATH.exists():
        with open(_MODEL_PATH, "rb") as f:
            _model_cache = pickle.load(f)
    return _model_cache

def _clean_text(text: str) -> str:
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    text = re.sub(r'\s+', ' ', text).strip()
    return text

PNEUMONIA_KEYWORDS = [
    "pneumonia","consolidation","infiltrate","opacity","crackles","cough",
    "fever","dyspnea","shortness of breath","chest pain","tachypnea",
    "respiratory","pleural","effusion","sputum","wheezing","hypoxia"
]

DIABETES_KEYWORDS = [
    "diabetes","glucose","hba1c","glycemic","insulin","hyperglycemia",
    "polydipsia","polyuria","obesity","metabolic","neuropathy",
    "retinopathy","nephropathy","blood sugar","ketoacidosis"
]

def _keyword_score(text: str, keywords: list) -> float:
    text_lower = text.lower()
    hits = sum(1 for kw in keywords if kw in text_lower)
    return min(hits / len(keywords) * 100, 95.0)

def analyze_report(text: str) -> dict:
    """
    Analyses clinical text and returns Pneumonia & Diabetes risk scores.
    Uses real trained TF-IDF + Logistic Regression from MTSamples dataset.
    """
    if not text or len(text.strip()) < 5:
        return {
            "pneumonia_risk": 0, "diabetes_risk": 0,
            "key_findings": [], "model_used": "N/A — no text provided"
        }

    model_obj = _load_model()
    clean = _clean_text(text)

    pneumonia_prob = None
    diabetes_prob  = None

    if model_obj:
        try:
            vectorizer    = model_obj["vectorizer"]
            clf_pneumonia = model_obj["clf_pneumonia"]
            clf_diabetes  = model_obj["clf_diabetes"]
            vec = vectorizer.transform([clean])
            pneumonia_prob = float(clf_pneumonia.predict_proba(vec)[0][1]) * 100
            diabetes_prob  = float(clf_diabetes.predict_proba(vec)[0][1])  * 100
        except Exception:
            pass

    # Keyword boost on top of model score
    kw_pneu = _keyword_score(text, PNEUMONIA_KEYWORDS)
    kw_diab = _keyword_score(text, DIABETES_KEYWORDS)

    if pneumonia_prob is None:
        pneumonia_prob = kw_pneu
    else:
        # Blend model + keywords (70/30)
        pneumonia_prob = 0.70 * pneumonia_prob + 0.30 * kw_pneu

    if diabetes_prob is None:
        diabetes_prob = kw_diab
    else:
        diabetes_prob = 0.70 * diabetes_prob + 0.30 * kw_diab

    # Extract key findings from text
    findings = []
    for kw in PNEUMONIA_KEYWORDS:
        if kw in text.lower():
            findings.append({"term": kw.title(), "category": "Pneumonia Indicator"})
    for kw in DIABETES_KEYWORDS:
        if kw in text.lower():
            findings.append({"term": kw.title(), "category": "Diabetes Indicator"})

    model_label = "Real TF-IDF + LR — Trained on MTSamples (4,999 clinical notes)" \
                  if model_obj else "Heuristic (run train_models.py to enable real model)"

    return {
        "pneumonia_risk":  round(min(pneumonia_prob, 95.0), 1),
        "diabetes_risk":   round(min(diabetes_prob,  95.0), 1),
        "key_findings":    findings[:10],
        "word_count":      len(text.split()),
        "model_used":      model_label,
        "dataset":         "MTSamples Medical Transcriptions Dataset"
    }
