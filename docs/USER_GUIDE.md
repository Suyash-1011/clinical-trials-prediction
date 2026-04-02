# Clinical Trials Outcome Prediction - User Guide

## Overview
This guide explains how to use the modular ML pipeline for predicting clinical trial outcomes. The pipeline supports data cleaning, feature engineering, model training, evaluation, and prediction via CLI scripts.

## Prerequisites
- Python 3.8+
- All dependencies in requirements.txt installed (see below)

## Installation
```sh
pip install -r requirements.txt
```

## Directory Structure
- `src/`: Core package code
- `scripts/`: CLI tools
- `data/`: Place your input CSV files here
- `images&graphs/`: Output figures and plots
- `tests/`: Unit tests
- `docs/`: Documentation

## CLI Usage

### 1. Data Cleaning
```sh
python scripts/data_cli.py data/raw.csv --output data/cleaned.csv
```

### 2. Model Training
```sh
python scripts/train_cli.py data/cleaned.csv Target --model rf --output model.joblib
```

### 3. Model Evaluation
```sh
python scripts/eval_cli.py data/cleaned.csv Target --model model.joblib
```

### 4. Prediction
```sh
python scripts/predict_cli.py data/new_samples.csv --model model.joblib --output predictions.csv
```

### 5. End-to-End Pipeline
```sh
python scripts/pipeline_cli.py data/raw.csv Target --model rf --output_model model.joblib --output_metrics metrics.txt
```

## Advanced Features
- Feature selection, scaling, and PCA are available in src/features.py
- Model types: rf, svm, xgb, catboost, logreg
- See README.md for more details

## Troubleshooting
- Ensure all dependencies are installed
- For XGBoost on macOS, install libomp: `brew install libomp`

## Contact
For issues, see the project README or open an issue on GitHub.
