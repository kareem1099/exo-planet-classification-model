# Exoplanet Classification: Real Planet or False Positive?

Binary classifier on NASA's Kepler Objects of Interest (KOI) table that separates **confirmed planets** from **false positives** using only the physical properties of the transit and its host star, then ranks the open **candidates** by how likely they are to be real planets.

## Results (test set, unseen stars)

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| **Gradient Boosting** (chosen by grouped CV) | **93.6%** | **0.926** | **0.982** |

Cross-validation ROC-AUC on the training set: Gradient Boosting 0.978, Random Forest 0.976, Logistic Regression 0.943.

Stable across 5 different star-based splits (92.9%–94.0% accuracy).
Of 1,369 open candidates, 265 get a probability above 0.9 of being a real planet.

## What v2 fixes

The first version reported 95%+ accuracy, but the setup leaked the answer. v2 removes three leaks:

1. **NASA vetting flags** (`koi_fpflag_*`): 98% of false positives have one and almost no confirmed planet does. They are NASA's decision, not a measurement, so they are removed.
2. **Same star in train and test**: one star can have several signals with identical star values. v2 splits **by star** (`kepid`) and uses grouped cross-validation.
3. **Missing values**: some measurements are missing far more often for false positives (missing not at random), so imputing would still let the model learn "missing = false positive". v2 uses complete rows only.

It also picks the model with cross-validation on the training set and uses the test set **once**. For comparison, a v1-style setup (flags + random split) reaches 99.7% on the same data.

## Data

NASA Exoplanet Archive, KOI cumulative table (9,564 signals). The notebook downloads it directly from:
`https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+cumulative&format=csv`

## Run

```bash
pip install pandas numpy scikit-learn matplotlib
jupyter notebook exoplanet_classification_v2.ipynb
```

## Limitations

- Only signals with all 22 measurements are used (6,619 of 7,587 labelled signals), so results apply to that group.
- `app.py` is the original v1 pipeline, kept for reference.
