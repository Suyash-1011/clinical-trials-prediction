#!/usr/bin/env python3
"""
Prediction CLI for loading model bundles and making predictions.

Usage:
    python scripts/predict_cli.py \
        --input data/test.csv \
        --model_bundle model.joblib \
        --output predictions.csv

Model bundles include: model, scaler, and columns to keep.
This ensures the same preprocessing is applied at inference time.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import pandas as pd
import joblib
from src import data as data_module
from src import features, models, utils


def main():
	parser = argparse.ArgumentParser(
		description='Make predictions with a trained model bundle'
	)
	parser.add_argument('--input', required=True,
						help='Input CSV file for prediction')
	parser.add_argument('--model_bundle', required=True,
						help='Path to saved model bundle (.joblib)')
	parser.add_argument('--output', required=True,
						help='Output CSV file for predictions and probabilities')
	args = parser.parse_args()

	logger = utils.get_logger('predict')

	# --- Load model bundle ---
	logger.info(f"Loading model bundle from: {args.model_bundle}")
	bundle = joblib.load(args.model_bundle)
	model = bundle['model']
	scaler = bundle['scaler']
	cols_to_keep = bundle['cols_to_keep']
	feature_names_after_eng = bundle.get('feature_names_after_engineering', None)

	# --- Load and clean data ---
	logger.info(f"Loading data from: {args.input}")
	df = data_module.load_and_clean(args.input)
	X = df.copy()

	# --- Apply feature engineering ---
	logger.info("Applying feature engineering...")
	X = features.engineer_features(X)

	# --- Scale using fitted scaler ---
	logger.info("Scaling features using fitted scaler...")
	X_scaled = scaler.transform(X)
	X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

	# --- Keep only the columns that were kept during training ---
	logger.info(f"Selecting {len(cols_to_keep)} features used during training...")
	X_final = X_scaled[cols_to_keep]

	# --- Make predictions ---
	logger.info("Making predictions...")
	y_pred, y_prob = models.predict(model, X_final)

	# --- Save predictions ---
	df_out = df.copy()
	df_out['prediction'] = y_pred
	if y_prob is not None:
		df_out['probability'] = y_prob

	utils.ensure_dir(os.path.dirname(os.path.abspath(args.output)))
	df_out.to_csv(args.output, index=False)
	logger.info(f"Predictions saved to: {args.output}")
	logger.info(f"Total predictions: {len(y_pred)}")
	logger.info(f"Positive class predictions: {(y_pred==1).sum()}, Negative: {(y_pred==0).sum()}")


if __name__ == '__main__':
	main()
