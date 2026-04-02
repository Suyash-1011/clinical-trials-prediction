import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#!/usr/bin/env python3
"""
CLI for model evaluation
"""
import argparse
from src import data, features, models, evaluate
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument('input', help='Input CSV file path')
    parser.add_argument('target', help='Target column name')
    parser.add_argument('--model', required=True, help='Trained model file path')
    args = parser.parse_args()
    df = data.load_and_clean(args.input)
    y = df[args.target]
    X = df.drop(columns=[args.target])
    X_scaled = features.scale_features(X)
    model = models.load_model(args.model)
    y_pred, y_prob = models.predict(model, X_scaled)
    metrics = evaluate.evaluate_classification(y, y_pred, y_prob)
    print("Evaluation metrics:")
    for k, v in metrics.items():
        if k not in ('confusion_matrix', 'report'):
            print(f"{k}: {v}")
    print("Confusion matrix:\n", metrics['confusion_matrix'])
    print("Classification report:\n", metrics['report'])

if __name__ == "__main__":
    main()
