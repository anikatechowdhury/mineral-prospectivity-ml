# mineral-prospectivity-ml

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

Machine learning-based mineral prospectivity prediction using Random Forest and Gradient Boosting classifiers. Compares data-driven ML predictions against traditional weighted overlay (MPI) to evaluate expert-weight vs learned-weight approaches for exploration targeting.

---

## Overview

Traditional mineral prospectivity mapping assigns expert-defined weights to evidence layers. This project replaces manual weighting with machine learning — training RF and Gradient Boosting classifiers on the same geological, structural, geochemical, and alteration evidence layers used in weighted overlay, then comparing the two approaches spatially and statistically.

The key research question: **do ML-learned feature importances agree with expert-assigned weights, and where do the two approaches differ?**

---

## Features

| Feature | Description |
|---|---|
| Lithology favourability | Mafic/ultramafic = high, sediment = low |
| Structural proximity | Distance decay from fault/shear zones |
| Hydrothermal alteration | Intensity of alteration halos |
| Geochemical anomaly | Stream sediment multi-element anomaly |
| Deposit proximity | Distance to known mineral occurrences |

---

## Models

| Model | CV Accuracy |
|---|---|
| Random Forest (200 trees) | ~97.5% |
| Gradient Boosting (150 estimators) | ~98.0% |

---

## Key Comparison — ML vs MPI Weights

| Layer | MPI Weight | RF Importance | GB Importance |
|---|---|---|---|
| Lithology | 30% | learned | learned |
| Structure | 25% | learned | learned |
| Alteration | 20% | learned | learned |
| Geochemistry | 15% | learned | learned |
| Proximity | 10% | learned | learned |

Feature importance plot directly overlays MPI weights vs ML-learned importances for each evidence layer.

---

## Project Structure

```
mineral-prospectivity-ml/
│
├── mineral_prospectivity_ml.py   # Main ML script
├── outputs/
│   ├── prospectivity_comparison.png
│   ├── feature_importance.png
│   ├── confusion_matrices.png
│   ├── roc_curves.png
│   ├── cv_comparison.png
│   ├── classification_report_RF.csv
│   └── classification_report_GB.csv
└── README.md
```

---

## Dependencies

| Library | Purpose |
|---|---|
| `numpy` | Feature grid generation |
| `pandas` | Report export |
| `matplotlib` | All visualisation |
| `scikit-learn` | RF, GB, CV, metrics, ROC |
| `scipy` | Spatial layer generation |

```bash
pip install numpy pandas matplotlib scikit-learn scipy
```

---

## Usage

```bash
python mineral_prospectivity_ml.py
```

---

## Outputs

| File | Description |
|---|---|
| `prospectivity_comparison.png` | Side-by-side MPI, RF, GB prospectivity maps |
| `feature_importance.png` | ML feature importance vs MPI expert weights |
| `confusion_matrices.png` | RF and GB confusion matrices |
| `roc_curves.png` | Per-class ROC curves for both models |
| `cv_comparison.png` | 5-fold CV accuracy — RF vs GB |
| `classification_report_*.csv` | Precision, recall, F1 per class |

---

## Author
Anikate Chowdhury  
ORCID: https://orcid.org/0009-0004-5580-2470

---

## Citation
If you use this methodology or implementation logic in academic or technical work,
please cite this repository. 

DOI: 
