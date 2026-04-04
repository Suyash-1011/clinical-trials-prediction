#!/usr/bin/env python3
"""
End-to-end pipeline CLI: data cleaning, feature engineering, feature selection,
model training, evaluation, and saving.

Usage:
    python scripts/pipeline_cli.py \
        --train data/dat_mol_desc.csv \
        --val data/val_mol_desc.csv \
        --target Target \
        --model catboost \
        --output_model model.joblib \
        --output_metrics metrics.txt

The pipeline follows the paper's methodology:
  1. Load pre-split train and validation CSVs (time-split already applied)
  2. Clean data (deduplicate, impute)
  3. Engineer interaction features (descriptor datasets only)
  4. Scale features (StandardScaler fit on train, transform both)
  5. Remove correlated features (threshold=0.9, fit on train only)
  6. Train model on processed train set
  7. Evaluate on processed validation set
  8. Save model and metrics
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from src import data as data_module
from src import features, models, evaluate, utils


def fit_transform_scaler(X_train):
	"""Fit StandardScaler on train, return fitted scaler and transformed train."""
	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(X_train)
	return pd.DataFrame(X_scaled, columns=X_train.columns, index=X_train.index), scaler


def transform_with_scaler(X, scaler):
	"""Apply a pre-fitted scaler to X."""
	X_scaled = scaler.transform(X)
	return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)


def fit_remove_correlated(X_train, threshold=0.9):
	"""Identify correlated columns on train set. Return columns to keep."""
	corr_matrix = X_train.corr().abs()
	upper = corr_matrix.where(
		np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
	)
	to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
	cols_to_keep = [col for col in X_train.columns if col not in to_drop]
	return cols_to_keep, to_drop


def main():
	parser = argparse.ArgumentParser(
		description='End-to-end ML pipeline for clinical trial outcome prediction'
	)
	parser.add_argument('--train', required=True,
						help='Training CSV file (e.g. dat_mol_desc.csv or dat_*_1024FP.csv)')
	parser.add_argument('--val', required=True,
						help='Validation CSV file (e.g. val_mol_desc.csv or val_*_1024FP.csv)')
	parser.add_argument('--target', required=True,
						help='Target column name (e.g. Target)')
	parser.add_argument('--model',
						choices=['rf', 'svm', 'xgb', 'catboost', 'logreg'],
						required=True,
						help='Model type to train')
	parser.add_argument('--output_model', required=True,
						help='Output path to save the trained model (.joblib)')
	parser.add_argument('--output_metrics', required=True,
						help='Output path to save metrics (.txt)')
	parser.add_argument('--corr_threshold', type=float, default=0.9,
						help='Pearson correlation threshold for feature removal (default 0.9)')
	parser.add_argument('--seed', type=int, default=42,
						help='Random seed for reproducibility')
	args = parser.parse_args()

	utils.set_seed(args.seed)
	logger = utils.get_logger('pipeline')

	# --- Step 1: Load and clean ---
	logger.info(f"Loading training data from: {args.train}")
	df_train = data_module.load_and_clean(args.train)
	logger.info(f"Loading validation data from: {args.val}")
	df_val = data_module.load_and_clean(args.val)

	logger.info(f"Train shape: {df_train.shape}, Val shape: {df_val.shape}")

	y_train = df_train[args.target]
	X_train = df_train.drop(columns=[args.target])

	y_val = df_val[args.target]
	X_val = df_val.drop(columns=[args.target])

	logger.info(f"Train class distribution: {y_train.value_counts().to_dict()}")
	logger.info(f"Val class distribution: {y_val.value_counts().to_dict()}")

	# --- Step 2: Feature engineering ---
	logger.info("Applying feature engineering...")
	X_train = features.engineer_features(X_train)
	X_val = features.engineer_features(X_val)
	logger.info(f"Feature matrix shape after engineering: {X_train.shape}")

	# --- Step 3: Scale features (fit on train only) ---
	logger.info("Scaling features (StandardScaler fit on train)...")
	X_train_scaled, scaler = fit_transform_scaler(X_train)
	X_val_scaled = transform_with_scaler(X_val, scaler)

	# --- Step 4: Remove correlated features (fit on train only) ---
	logger.info(f"Removing correlated features (threshold={args.corr_threshold})...")
	cols_to_keep, dropped_cols = fit_remove_correlated(X_train_scaled, threshold=args.corr_threshold)
	X_train_final = X_train_scaled[cols_to_keep]
	X_val_final = X_val_scaled[cols_to_keep]
	logger.info(f"Dropped {len(dropped_cols)} correlated features. Remaining: {len(cols_to_keep)}")
	if dropped_cols:
		logger.info(f"Dropped columns: {dropped_cols[:10]}{'...' if len(dropped_cols)>10 else ''}")

	# --- Step 5: Train model ---
	logger.info(f"Training {args.model} model on {X_train_final.shape[0]} samples, {X_train_final.shape[1]} features...")
	model = models.train_model(X_train_final, y_train, model_type=args.model, random_state=args.seed)

	# --- Step 6: Evaluate ---
	logger.info("Evaluating on validation set...")
	y_pred, y_prob = models.predict(model, X_val_final)
	metrics = evaluate.evaluate_classification(y_val, y_pred, y_prob)

	# --- Step 7: Save model ---
	# Save model bundle: model + scaler + kept columns list
	# This ensures the same preprocessing is applied at inference time
	model_bundle = {
		'model': model,
		'scaler': scaler,
		'cols_to_keep': cols_to_keep,
		'feature_names_after_engineering': list(X_train.columns),
	}
	utils.ensure_dir(os.path.dirname(os.path.abspath(args.output_model)))
	joblib.dump(model_bundle, args.output_model)
	logger.info(f"Model bundle saved to: {args.output_model}")

	# --- Step 8: Save metrics ---
	utils.ensure_dir(os.path.dirname(os.path.abspath(args.output_metrics)))
	with open(args.output_metrics, 'w') as f:
		for k, v in metrics.items():
			if k not in ('confusion_matrix', 'report'):
				f.write(f"{k}: {v}\n")
		f.write(f"Confusion matrix:\n{metrics['confusion_matrix']}\n")
		f.write(f"Classification report:\n{metrics['report']}\n")
		f.write(f"\nPipeline settings:\n")
		f.write(f"  model_type: {args.model}\n")
		f.write(f"  train_file: {args.train}\n")
		f.write(f"  val_file: {args.val}\n")
		f.write(f"  corr_threshold: {args.corr_threshold}\n")
		f.write(f"  features_after_engineering: {X_train.shape[1]}\n")
		f.write(f"  features_after_corr_removal: {len(cols_to_keep)}\n")

	logger.info(f"Metrics saved to: {args.output_metrics}")
	logger.info(f"AUC: {metrics.get('auc', 'N/A'):.4f}")
	logger.info(f"MCC: {metrics.get('mcc', 'N/A'):.4f}")
	logger.info("Pipeline complete.")


if __name__ == '__main__':
	main()
