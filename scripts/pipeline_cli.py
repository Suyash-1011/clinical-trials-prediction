import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#!/usr/bin/env python3
"""
End-to-end pipeline CLI: data cleaning, feature engineering, model training, evaluation, and prediction.
"""
import argparse
from src import data, features, models, evaluate, utils
import pandas as pd
import os


import argparse
import pandas as pd
from src import data, features, models, evaluate

def main():
    parser = argparse.ArgumentParser(description='End-to-end ML pipeline for clinical trial outcome prediction')
    parser.add_argument('input', help='Input CSV file')
    parser.add_argument('target', help='Target column name')
    parser.add_argument('--model', choices=['rf', 'svm', 'xgb', 'catboost', 'logreg'], required=True, help='Model type')
    parser.add_argument('--output_model', required=True, help='Output model file path')
    parser.add_argument('--output_metrics', required=True, help='Output metrics/report file path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set proportion (default 0.2)')
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    X = df.drop(columns=[args.target])
    y = df[args.target]

    # Split into train/test (stratified, random by default)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.seed, stratify=y)

    # Feature engineering (optional: add more steps)
    X_train = features.engineer_features(X_train)
    X_test = features.engineer_features(X_test)

    # Train model only on train set
    model = models.train_model(X_train, y_train, model_type=args.model)
    models.save_model(model, args.output_model)

    # Evaluate on test set only
    y_pred, y_prob = models.predict(model, X_test)
    metrics = evaluate.evaluate_classification(y_test, y_pred, y_prob)
    with open(args.output_metrics, 'w') as f:
        for k, v in metrics.items():
            if k != 'confusion_matrix' and k != 'report':
                f.write(f"{k}: {v}\n")
        f.write(f"Confusion matrix:\n{metrics['confusion_matrix']}\n")
        f.write(f"Classification report:\n{metrics['report']}\n")
    print(f"Pipeline complete. Model: {args.output_model}, Metrics: {args.output_metrics}")

if __name__ == '__main__':
    main()
