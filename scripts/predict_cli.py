import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#!/usr/bin/env python3
"""
CLI for making predictions with a trained model
"""
import argparse
from src import data, features, models
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Predict with a trained model.")
    parser.add_argument('input', help='Input CSV file path')
    parser.add_argument('--model', required=True, help='Trained model file path')
    parser.add_argument('--output', required=True, help='Output CSV file for predictions')
    args = parser.parse_args()
    df = data.load_and_clean(args.input)
    X_scaled = features.scale_features(df)
    model = models.load_model(args.model)
    y_pred, y_prob = models.predict(model, X_scaled)
    df_out = df.copy()
    df_out['prediction'] = y_pred
    if y_prob is not None:
        df_out['probability'] = y_prob
    df_out.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")

if __name__ == "__main__":
    main()
