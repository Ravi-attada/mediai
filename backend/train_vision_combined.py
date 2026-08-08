"""
MTech Project — Combined Vision Model Training Script
Combines CheXpert + NIH Chest X-rays datasets (335,534 total records)
to train a stronger Random Forest for Pneumonia detection.

Run:  python train_vision_combined.py
Output: trained_models/vision_model.pkl  (replaces previous single-dataset model)
"""

import os, sys, warnings, pickle
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path

OUT       = Path(__file__).parent / "trained_models"
OUT.mkdir(exist_ok=True)
DATA_ROOT = Path(r"C:\Mtech Project")

print("=" * 65)
print("  Combined Vision Model Training: CheXpert + NIH Chest X-rays")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────────────────
# DATASET 1: CheXpert  (223,414 rows, 14 binary label columns)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/4] Loading CheXpert dataset...")
df_cx = pd.read_csv(DATA_ROOT / "CheXpert-v1.0-small" / "train.csv")

CHEX_LABEL_COLS = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
    'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
    'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
    'Pleural Other', 'Fracture', 'Support Devices'
]

# Fill NaN with 0; clip negatives (uncertain = 0 for this baseline)
df_cx[CHEX_LABEL_COLS] = df_cx[CHEX_LABEL_COLS].fillna(0).clip(lower=0)

# Pneumonia label: 1 if Pneumonia column > 0
chex_y = (df_cx['Pneumonia'].fillna(0).clip(lower=0) > 0).astype(int)
chex_X = df_cx[CHEX_LABEL_COLS].values

print(f"   CheXpert: {len(df_cx):,} records  |  Pneumonia positive: {chex_y.sum():,}")

# ─────────────────────────────────────────────────────────────────────────────
# DATASET 2: NIH Chest X-rays  (112,120 rows, multi-label string column)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/4] Loading NIH Chest X-rays dataset...")
df_nih = pd.read_csv(DATA_ROOT / "NIH Chest X-rays" / "Data_Entry_2017.csv")

# Parse multi-label string into binary columns matching CheXpert features
NIH_MAPPING = {
    'No Finding':                 'No Finding',
    'Cardiomegaly':               'Cardiomegaly',
    'Effusion':                   'Pleural Effusion',
    'Infiltration':               'Lung Opacity',        # closest match
    'Nodule':                     'Lung Lesion',
    'Pneumonia':                  'Pneumonia',
    'Atelectasis':                'Atelectasis',
    'Pneumothorax':               'Pneumothorax',
    'Consolidation':              'Consolidation',
    'Edema':                      'Edema',
    'Emphysema':                  'Lung Lesion',
    'Fibrosis':                   'Lung Opacity',
    'Pleural_Thickening':         'Pleural Other',
    'Mass':                       'Lung Lesion',
}

# Build binary feature matrix for NIH matching CheXpert's label columns
nih_features = {col: np.zeros(len(df_nih)) for col in CHEX_LABEL_COLS}

for i, labels_str in enumerate(df_nih['Finding Labels'].astype(str)):
    labels = [l.strip() for l in labels_str.split('|')]
    for nih_label in labels:
        chex_col = NIH_MAPPING.get(nih_label)
        if chex_col and chex_col in nih_features:
            nih_features[chex_col][i] = 1.0
        if nih_label == 'No Finding':
            nih_features['No Finding'][i] = 1.0

nih_X = np.column_stack([nih_features[col] for col in CHEX_LABEL_COLS])
nih_y = (nih_features.get('Pneumonia', np.zeros(len(df_nih))) > 0).astype(int)

# Also check "Pneumonia" directly in string label
nih_y = np.array([1 if 'Pneumonia' in str(s) else 0
                  for s in df_nih['Finding Labels']], dtype=int)

print(f"   NIH:      {len(df_nih):,} records  |  Pneumonia positive: {nih_y.sum():,}")

# ─────────────────────────────────────────────────────────────────────────────
# COMBINE: Stack both datasets
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/4] Combining datasets...")
X_combined = np.vstack([chex_X, nih_X])
y_combined = np.concatenate([chex_y.values, nih_y])

print(f"   COMBINED: {len(X_combined):,} total records")
print(f"   Pneumonia positive: {y_combined.sum():,} ({y_combined.mean()*100:.1f}%)")
print(f"   Feature columns: {CHEX_LABEL_COLS}")

# ─────────────────────────────────────────────────────────────────────────────
# TRAIN: Random Forest on combined dataset
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/4] Training combined Random Forest model...")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y_combined, test_size=0.2, random_state=42, stratify=y_combined
)

clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    n_jobs=-1,
    random_state=42,
    class_weight='balanced'   # handles Pneumonia class imbalance
)
clf.fit(X_train, y_train)

y_prob = clf.predict_proba(X_test)[:, 1]
y_pred = clf.predict(X_test)
auc    = roc_auc_score(y_test, y_prob)

print(f"\n   AUC-ROC:  {auc:.4f}")
print(classification_report(y_test, y_pred, target_names=["No Pneumonia", "Pneumonia"]))

# Feature importance
importances = dict(zip(CHEX_LABEL_COLS, clf.feature_importances_))
top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
print("   Top 5 Most Predictive Features:")
for feat, imp in top_features:
    print(f"     {feat:<35} {imp:.4f}")

# Save
with open(OUT / "vision_model.pkl", "wb") as f:
    pickle.dump({
        "clf":          clf,
        "feature_cols": CHEX_LABEL_COLS,
        "datasets":     ["CheXpert-v1.0-small", "NIH-Chest-X-rays"],
        "total_records": len(X_combined),
        "auc":          auc
    }, f)

print(f"\n   Saved -> trained_models/vision_model.pkl")

print("\n" + "=" * 65)
print("  COMBINED VISION MODEL TRAINING COMPLETE!")
print("=" * 65)
print(f"  CheXpert records  : {len(df_cx):,}")
print(f"  NIH records       : {len(df_nih):,}")
print(f"  Total combined    : {len(X_combined):,}")
print(f"  Final AUC-ROC     : {auc:.4f}")
print("=" * 65)
