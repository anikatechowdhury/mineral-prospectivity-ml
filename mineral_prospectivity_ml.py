"""
Mineral Prospectivity Prediction Using Machine Learning
--------------------------------------------------------
Predicts mineral prospectivity zones using Random Forest
and XGBoost classifiers trained on geological, structural,
geochemical and alteration evidence layers.

Compares ML-based prediction against traditional weighted
overlay (MPI) to evaluate data-driven vs expert-weight approaches.

Evidence Features:
  - Lithology favourability
  - Structural proximity
  - Hydrothermal alteration
  - Geochemical anomaly
  - Deposit proximity

Target:
  - Prospectivity class (Low / Moderate / High / Very High)
  - Derived from MPI weighted overlay (ground truth)

Outputs:
  - RF and XGBoost classification
  - Feature importance comparison
  - Predicted prospectivity maps
  - ML vs MPI agreement analysis
  - ROC curves
  - Confusion matrices

Context: Metalliferous terrain — Singhbhum Craton /
Proterozoic fold belt setting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.ndimage import gaussian_filter, distance_transform_edt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                              ConfusionMatrixDisplay, roc_curve, auc)
from sklearn.preprocessing import label_binarize
import warnings
import os

warnings.filterwarnings("ignore")
np.random.seed(88)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ROWS, COLS = 120, 120

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — REPRODUCE EVIDENCE LAYERS (same as GIS repo)
# ═══════════════════════════════════════════════════════════════════════════════

# Lithology
litho_base = np.zeros((ROWS, COLS))
litho_base[30:80, 20:70] = 8.0
litho_base[10:45, 65:110] = 5.0
litho_base[70:120, 0:50] = 2.0
litho_base[litho_base == 0] = 3.5
lithology = np.clip(gaussian_filter(litho_base, sigma=8)
                    + 0.8 * np.random.randn(ROWS, COLS), 1, 10)

# Structure proximity
struct_mask = np.zeros((ROWS, COLS))
for i in range(ROWS):
    j = int(i * COLS / ROWS)
    if j < COLS:
        struct_mask[i, max(0, j-2):min(COLS, j+2)] = 1
struct_mask[55:58, 20:90] = 1
dist = distance_transform_edt(1 - struct_mask)
structure = np.clip(10 * (1 - dist/dist.max())
                    + 0.5 * np.random.randn(ROWS, COLS), 1, 10)
structure = gaussian_filter(structure, sigma=6)

# Alteration
alt_base = np.zeros((ROWS, COLS))
for (cy, cx) in [(55, 55), (35, 35), (75, 80)]:
    yy, xx = np.ogrid[:ROWS, :COLS]
    rr = np.sqrt((yy-cy)**2 + (xx-cx)**2)
    alt_base += 8 * np.exp(-rr**2 / (2*18**2))
alteration = np.clip(gaussian_filter(alt_base, sigma=5)
                     + 0.6 * np.random.randn(ROWS, COLS), 1, 10)

# Geochemistry
geochem_base = 3 + 1.5 * np.random.randn(ROWS, COLS)
for (cy, cx, amp) in [(45, 45, 6), (60, 70, 5), (30, 80, 4)]:
    yy, xx = np.ogrid[:ROWS, :COLS]
    rr = np.sqrt((yy-cy)**2 + (xx-cx)**2)
    geochem_base += amp * np.exp(-rr**2 / (2*15**2))
geochemistry = np.clip(gaussian_filter(geochem_base, sigma=7), 1, 10)

# Deposit proximity
deposit_pts  = [(40, 40), (65, 62), (28, 75)]
deposit_locs = np.zeros((ROWS, COLS))
for (dy, dx) in deposit_pts:
    deposit_locs[dy, dx] = 1
dist_dep = distance_transform_edt(1 - deposit_locs)
proximity = np.clip(10 * (1 - dist_dep/dist_dep.max())
                    + 0.4 * np.random.randn(ROWS, COLS), 1, 10)
proximity = gaussian_filter(proximity, sigma=10)

WEIGHTS = {"lithology": 0.30, "structure": 0.25,
           "alteration": 0.20, "geochemistry": 0.15,
           "proximity": 0.10}

layers = {"lithology": lithology, "structure": structure,
          "alteration": alteration, "geochemistry": geochemistry,
          "proximity": proximity}

MPI = sum(WEIGHTS[k] * layers[k] for k in WEIGHTS)

# MPI classes
def classify_mpi(arr):
    cls = np.zeros_like(arr, dtype=int)
    cls[arr < 3]  = 0   # Low
    cls[(arr >= 3) & (arr < 5)] = 1   # Moderate
    cls[(arr >= 5) & (arr < 7)] = 2   # High
    cls[arr >= 7] = 3   # Very High
    return cls

MPI_class = classify_mpi(MPI)

CLASS_NAMES  = ["Low", "Moderate", "High", "Very High"]
CLASS_COLORS = {0: "#E8F5E9", 1: "#FFF9C4",
                2: "#FF8F00", 3: "#B71C1C"}

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — PREPARE ML DATASET
# ═══════════════════════════════════════════════════════════════════════════════

print("── Preparing ML Dataset ──")

# Flatten spatial grids to tabular format
feature_names = list(layers.keys())
X = np.column_stack([layers[k].flatten() for k in feature_names])
y = MPI_class.flatten()

print(f"  Dataset: {X.shape[0]} samples × {X.shape[1]} features")
print(f"  Class distribution: {dict(zip(CLASS_NAMES, [int((y==i).sum()) for i in range(4)]))}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)
print(f"  Train: {len(X_train)}  Test: {len(X_test)}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — RANDOM FOREST
# ═══════════════════════════════════════════════════════════════════════════════

print("\n── Random Forest ──")
rf = RandomForestClassifier(n_estimators=200, max_depth=None,
                             class_weight="balanced",
                             random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf   = rf.predict(X_test)
y_prob_rf   = rf.predict_proba(X_test)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_rf = cross_val_score(rf, X, y, cv=cv, scoring="accuracy")
print(f"  CV Accuracy: {cv_rf.mean():.4f} ± {cv_rf.std():.4f}")
print(classification_report(y_test, y_pred_rf,
                              target_names=CLASS_NAMES))

# Full map prediction
rf_map = rf.predict(X).reshape(ROWS, COLS)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — GRADIENT BOOSTING (XGBoost-style)
# ═══════════════════════════════════════════════════════════════════════════════

print("── Gradient Boosting ──")
gb = GradientBoostingClassifier(n_estimators=150, max_depth=4,
                                 learning_rate=0.1, random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
y_prob_gb = gb.predict_proba(X_test)

cv_gb = cross_val_score(gb, X, y, cv=cv, scoring="accuracy")
print(f"  CV Accuracy: {cv_gb.mean():.4f} ± {cv_gb.std():.4f}")
print(classification_report(y_test, y_pred_gb,
                              target_names=CLASS_NAMES))

gb_map = gb.predict(X).reshape(ROWS, COLS)

# ═══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

cmap_p = ListedColormap([CLASS_COLORS[k] for k in range(4)])
bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
norm   = BoundaryNorm(bounds, cmap_p.N)

patches = [mpatches.Patch(color=CLASS_COLORS[k],
                           label=CLASS_NAMES[k])
           for k in range(4)]

# ── Plot 1: MPI vs RF vs GB maps ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
titles = ["MPI Weighted Overlay\n(Traditional)",
          "Random Forest\n(ML Prediction)",
          "Gradient Boosting\n(ML Prediction)"]
maps   = [MPI_class, rf_map, gb_map]

for ax, title, mp in zip(axes, titles, maps):
    ax.imshow(mp, cmap=cmap_p, norm=norm)
    for (dy, dx) in deposit_pts:
        ax.plot(dx, dy, "k*", markersize=10, zorder=5)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axis("off")

axes[0].legend(handles=patches, loc="lower right",
               fontsize=8, framealpha=0.9,
               title="Prospectivity")
fig.suptitle("Mineral Prospectivity — MPI vs ML Predictions\n"
             "(★ = Known deposit locations)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/prospectivity_comparison.png",
            dpi=180, bbox_inches="tight")
plt.close()
print("\nSaved: prospectivity_comparison.png")

# ── Plot 2: Feature importance comparison ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, model, title, col in zip(
        axes,
        [rf, gb],
        ["Random Forest — Feature Importance",
         "Gradient Boosting — Feature Importance"],
        ["#1565C0", "#B71C1C"]):
    imp = pd.Series(model.feature_importances_,
                    index=feature_names).sort_values(ascending=True)
    bars = ax.barh(imp.index, imp.values,
                   color=col, alpha=0.8, edgecolor="white")
    # MPI weight overlay
    mpi_w = pd.Series(WEIGHTS).reindex(imp.index)
    ax.scatter(mpi_w.values, range(len(imp)),
               color="red", zorder=5, s=60,
               label="MPI weight (normalised)")
    for bar, val in zip(bars, imp.values):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=8)
    ax.set_xlabel("Feature Importance (Gini)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle("ML Feature Importance vs MPI Expert Weights",
             fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/feature_importance.png",
            dpi=180, bbox_inches="tight")
plt.close()
print("Saved: feature_importance.png")

# ── Plot 3: Confusion matrices ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 6))

for ax, y_pred, title, cmap in zip(
        axes,
        [y_pred_rf, y_pred_gb],
        ["Random Forest", "Gradient Boosting"],
        ["Blues", "Reds"]):
    cm   = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, cmap=cmap, colorbar=False,
              text_kw={"fontsize": 10, "fontweight": "bold"})
    ax.set_title(f"Confusion Matrix — {title}",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("True", fontsize=9)

plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/confusion_matrices.png",
            dpi=180, bbox_inches="tight")
plt.close()
print("Saved: confusion_matrices.png")

# ── Plot 4: ROC curves — RF ───────────────────────────────────────────────────
y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
roc_colors = [CLASS_COLORS[k] for k in range(4)]
# Fix very light color for Low class
roc_colors[0] = "#2E7D32"

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, y_prob, title in zip(
        axes,
        [y_prob_rf, y_prob_gb],
        ["Random Forest — ROC Curves",
         "Gradient Boosting — ROC Curves"]):
    for i, (cls, col) in enumerate(zip(CLASS_NAMES, roc_colors)):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=col, linewidth=2,
                label=f"{cls}  AUC={roc_auc:.2f}")
    ax.plot([0,1],[0,1],"k--", linewidth=1, alpha=0.4)
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/roc_curves.png",
            dpi=180, bbox_inches="tight")
plt.close()
print("Saved: roc_curves.png")

# ── Plot 5: CV accuracy comparison ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
folds = [f"Fold {i+1}" for i in range(5)]
x     = np.arange(5)
w     = 0.35
ax.bar(x - w/2, cv_rf, width=w, color="#1565C0",
       alpha=0.85, label=f"RF  (mean={cv_rf.mean():.3f})",
       edgecolor="white")
ax.bar(x + w/2, cv_gb, width=w, color="#B71C1C",
       alpha=0.85, label=f"GB  (mean={cv_gb.mean():.3f})",
       edgecolor="white")
ax.axhline(cv_rf.mean(), color="#1565C0", linestyle="--",
           linewidth=1.3, alpha=0.7)
ax.axhline(cv_gb.mean(), color="#B71C1C", linestyle="--",
           linewidth=1.3, alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(folds)
ax.set_ylim(0, 1.1)
ax.set_ylabel("CV Accuracy", fontsize=10)
ax.set_title("5-Fold Cross-Validation — RF vs Gradient Boosting",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/cv_comparison.png",
            dpi=180, bbox_inches="tight")
plt.close()
print("Saved: cv_comparison.png")

# ── Save classification reports ───────────────────────────────────────────────
for model_name, y_pred in [("RF", y_pred_rf), ("GB", y_pred_gb)]:
    rpt = classification_report(y_test, y_pred,
                                 target_names=CLASS_NAMES,
                                 output_dict=True)
    pd.DataFrame(rpt).T.round(3).to_csv(
        f"{OUTPUT_DIR}/classification_report_{model_name}.csv")

print(f"\n✓ All outputs saved to: {OUTPUT_DIR}")
print(f"  RF CV Accuracy : {cv_rf.mean():.4f} ± {cv_rf.std():.4f}")
print(f"  GB CV Accuracy : {cv_gb.mean():.4f} ± {cv_gb.std():.4f}")
