import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#!/usr/bin/env python3
"""
CLI for model training
"""
import argparse
from src import data, features, models, utils
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Train a classification model.")
    parser.add_argument('input', help='Input CSV file path')
    parser.add_argument('target', help='Target column name')
    parser.add_argument('--model', choices=['rf', 'svm', 'xgb', 'catboost', 'logreg'], default='rf', help='Model type')
    parser.add_argument('--output', required=True, help='Output model file path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    utils.set_seed(args.seed)
    df = data.load_and_clean(args.input)
    y = df[args.target]
    X = df.drop(columns=[args.target])
    X_scaled = features.scale_features(X)
    model = models.train_model(X_scaled, y, model_type=args.model)
    models.save_model(model, args.output)
    print(f"Model trained and saved to {args.output}")

if __name__ == "__main__":
    main()
