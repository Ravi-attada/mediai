# 📂 Datasets

This folder contains the **tabular and text datasets** used in the MediAI project.

## ✅ Included Files (in this repo)

| File | Size | Used For |
|---|---|---|
| `diabetes.csv` | 23 KB | PIMA Diabetes — Tabular ML (blood glucose, HbA1c, BMI) |
| `mtsamples.csv` | 17 MB | MTSamples — NLP Module (clinical transcription notes) |
| `abstracts.txt` | 27 KB | Literature review abstracts (PubMed 2023–2025) |

---

## ❌ NOT Included — Download Required (Too Large for GitHub)

The X-ray image datasets are 70+ GB combined and cannot be stored on GitHub.
Download them from Kaggle and place them in your local `C:\Mtech Project\` folder:

### CheXpert v1.0-small — Stanford ML Group
- **Size:** ~28 GB | **Images:** 224,316 chest X-rays
- **Download:** https://www.kaggle.com/datasets/ashery/chexpert
- **Place at:** `C:\Mtech Project\CheXpert-v1.0-small\`

### NIH Chest X-rays — National Institutes of Health
- **Size:** ~45 GB | **Images:** 112,120 chest X-rays
- **Download:** https://www.kaggle.com/datasets/nih-chest-xrays/data
- **Place at:** `C:\Mtech Project\NIH Chest X-rays\`

---

## 📊 Combined Dataset Statistics

| Dataset | Records | Type | Source |
|---|---|---|---|
| CheXpert v1.0-small | 224,316 X-rays | Image | Stanford ML Group |
| NIH Chest X-rays | 112,120 X-rays | Image | NIH / Kaggle |
| **Combined Vision** | **336,436 X-rays** | Image | CheXpert + NIH |
| MTSamples | 4,999 records | Clinical text | Kaggle |
| PIMA Diabetes | 768 records | Structured (tabular) | Kaggle / UCI |

---

> **Note for Thesis Reviewers:** All datasets are publicly available and free to download.
> No private or patient-identifiable data was used in this research.
