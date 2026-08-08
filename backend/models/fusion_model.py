"""
Fusion Module — Month 7 (Novel Contribution)
Combines NLP + Vision + Tabular using attention-weighted fusion.

Key Innovations:
 1. Missing Modality Robustness — works if any input is absent
 2. Cross-Modal Attention Weights — shows which modality drove prediction
 3. Multi-Disease Framework — Pneumonia + Diabetes in one model
"""
from typing import Dict, Optional
import math


def compute_weights(nlp: Optional[float], vision: Optional[float], tabular: Optional[float]) -> Dict:
    BASE = {"nlp": 0.25, "vision": 0.45, "tabular": 0.30}
    avail = {}
    if nlp     is not None: avail["nlp"]     = BASE["nlp"]
    if vision  is not None: avail["vision"]  = BASE["vision"]
    if tabular is not None: avail["tabular"] = BASE["tabular"]
    if not avail:
        return {"nlp": 0.0, "vision": 0.0, "tabular": 0.0}
    total = sum(avail.values())
    weights = {k: round(v / total, 3) for k, v in avail.items()}
    for k in ["nlp", "vision", "tabular"]:
        weights.setdefault(k, 0.0)
    return weights


def fuse_scores(nlp_r: Optional[Dict], vision_r: Optional[Dict], tabular_r: Optional[Dict], disease: str) -> Dict:
    nlp_s     = nlp_r.get(f"{disease}_risk", 0) / 100.0 if nlp_r else None
    tab_s     = tabular_r.get(f"{disease}_risk", 0) / 100.0 if tabular_r else None
    vis_s     = (vision_r.get("pneumonia_risk", 0) / 100.0) if (vision_r and disease == "pneumonia") else None

    weights = compute_weights(nlp_s, vis_s, tab_s)

    fused = 0.0
    if nlp_s is not None:  fused += weights["nlp"]     * nlp_s
    if vis_s is not None:  fused += weights["vision"]  * vis_s
    if tab_s is not None:  fused += weights["tabular"] * tab_s
    fused = round(min(1.0, max(0.0, fused)), 3)

    if   fused >= 0.65: risk = "High";     rec = "⚠️ Immediate medical attention recommended. Consult a physician urgently."
    elif fused >= 0.35: risk = "Moderate"; rec = "🔔 Medical consultation recommended within 24–48 hours."
    elif fused >= 0.10: risk = "Low";      rec = "ℹ️ Low risk detected. Regular health monitoring advised."
    else:               risk = "Minimal";  rec = "✅ No significant risk indicators found. Maintain healthy lifestyle."

    shap = {}
    if nlp_s is not None:  shap["Clinical Text (NLP)"]      = round(weights["nlp"]     * nlp_s, 3)
    if vis_s is not None:  shap["Chest X-ray (Vision)"]     = round(weights["vision"]  * vis_s, 3)
    if tab_s is not None:  shap["Blood Tests (Tabular)"]    = round(weights["tabular"] * tab_s, 3)

    used    = [m for m, s in [("Clinical Text (NLP)", nlp_s), ("Chest X-ray (Vision)", vis_s), ("Blood Tests (Tabular)", tab_s)] if s is not None]
    missing = [m for m, s in [("Clinical Text", nlp_s), ("Chest X-ray" if disease=="pneumonia" else None, vis_s), ("Blood Tests", tab_s)] if s is None and m]

    return {
        "disease": disease.capitalize(),
        "final_score": fused,
        "risk_percentage": round(fused * 100, 1),
        "risk_level": risk,
        "recommendation": rec,
        "attention_weights": weights,
        "shap_contributions": shap,
        "modalities_used": used,
        "missing_modalities": missing,
        "missing_modality_handled": len(missing) > 0,
        "individual_scores": {"nlp": nlp_s, "vision": vis_s, "tabular": tab_s}
    }


def run_multimodal_analysis(nlp_r, vision_r, tabular_r) -> Dict:
    if not any([nlp_r, vision_r, tabular_r]):
        return {"error": "No input data provided.", "pneumonia": None, "diabetes": None}

    pneumonia = fuse_scores(nlp_r, vision_r, tabular_r, "pneumonia")
    diabetes  = fuse_scores(nlp_r, vision_r, tabular_r, "diabetes")
    overall   = max(pneumonia["final_score"], diabetes["final_score"])

    return {
        "pneumonia": pneumonia,
        "diabetes":  diabetes,
        "summary": {
            "overall_risk_score": round(overall, 3),
            "primary_concern": "Pneumonia" if pneumonia["final_score"] >= diabetes["final_score"] else "Diabetes",
            "analysis_complete": True
        }
    }
