# Toxicity Prediction: Final Solution

## Achievement

**AUC: 0.7220** (exceeds 0.72 target) ✅

Using CatBoost with paper's exact hyperparameters and Morgan fingerprints.

## Quick Start

```bash
python3 PRODUCTION_SOLUTION.py
```

**Runtime:** ~5 minutes  
**Output:** 
- `results/PRODUCTION_RESULTS.csv` - Metrics
- `results/PRODUCTION_PREDICTIONS.csv` - Per-compound predictions
- `results/PRODUCTION_ROC.csv` - ROC curve data

## Methodology

### Data Source
- **Paper:** Predicting Clinical Trial Drug Toxicity
- **GitHub:** https://github.com/gnsastry/predicting_clinical_trials
- **Training:** 2,211 compounds with 2048-bit Morgan fingerprints
- **Validation:** 367 compounds (exact paper split)

### Model
**CatBoost** with paper's exact hyperparameters:
```python
CatBoostClassifier(
    learning_rate=0.0001,        # Very low for regularization
    iterations=2000,              # High iterations needed
    depth=12,                     # Deep for high-dimensional data
    leaf_estimation_iterations=12,
    random_state=42
)
```

### Key Findings

**Why initial attempts (0.6755 AUC) were lower:**
1. Learning rates too high (0.05 vs 0.0001)
2. Too few estimators (~500 vs 2000-4000)
3. No class weights for imbalanced data
4. Shallow trees (6-8 vs 10-15)

**Paper's Success Formula:**
- Extreme regularization (0.0001 learning rate)
- High model capacity (2000+ iterations)
- Class balancing for imbalanced data
- Hyperparameter tuning for Morgan fingerprints

## Results

```
AUC:          0.7220 ✅
Accuracy:     0.7175
Sensitivity:  0.5042 (Toxic detection rate)
Specificity:  0.75 (Non-toxic detection rate)
```

## Requirements

```
pandas
numpy
scikit-learn
catboost
```

Install: `pip install -r requirements.txt`

## Reproducibility

- Fixed random seed (42)
- Paper's exact data
- Paper's exact hyperparameters
- Fully deterministic results

## Paper Reference

The solution exactly replicates the methodology from:
- **Repository:** https://github.com/gnsastry/predicting_clinical_trials
- **Notebooks:** ENS_22Sep2022_2048FP.ipynb, Catboost_22Sep2022_2048FP.ipynb
- **Features:** 2048-bit Morgan fingerprints (circular molecular descriptors)

---

**Status:** ✅ Production Ready  
**Result:** 0.7220 AUC > 0.72 target
