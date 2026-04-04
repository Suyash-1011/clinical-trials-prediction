#!/usr/bin/env python3
"""
Meta-ensemble training CLI: Train a two-level stacking ensemble combining
descriptor and fingerprint base models.

Usage:
    python scripts/meta_ensemble_cli.py \
        --train_desc data/dat_mol_desc.csv \
        --train_fp data/dat_*_1024FP.csv \
        --val_desc data/val_mol_desc.csv \
        --val_fp data/val_*_1024FP.csv \
        --target Target \
        --output_ensemble ensemble.joblib \
        --output_metrics ensemble_metrics.txt

The meta-ensemble:
  1. Trains CatBoost on descriptor features with out-of-fold CV
  2. Trains XGBoost on fingerprint features with out-of-fold CV
  3. Generates meta-features from both base models
  4. Trains LogisticRegression as meta-learner
  5. Saves the complete ensemble for inference
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import pandas as pd
from src import data as data_module
from src import features, models, evaluate, utils
from src.meta_ensemble import MetaEnsemble


def main():
	parser = argparse.ArgumentParser(
		description='Train a meta-ensemble combining descriptor and fingerprint models'
	)
	parser.add_argument('--train_desc', required=True,
						help='Training descriptor CSV file')
	parser.add_argument('--train_fp', required=True,
						help='Training fingerprint CSV file')
	parser.add_argument('--val_desc', required=True,
						help='Validation descriptor CSV file')
	parser.add_argument('--val_fp', required=True,
						help='Validation fingerprint CSV file')
	parser.add_argument('--target', required=True,
						help='Target column name')
	parser.add_argument('--output_ensemble', required=True,
						help='Output path to save the ensemble (.joblib)')
	parser.add_argument('--output_metrics', required=True,
						help='Output path to save metrics (.txt)')
	parser.add_argument('--seed', type=int, default=42,
						help='Random seed')
	args = parser.parse_args()

	utils.set_seed(args.seed)
	logger = utils.get_logger('meta_ensemble')

	# --- Load and clean data ---
	logger.info("Loading descriptor and fingerprint data...")
	df_train_desc = data_module.load_and_clean(args.train_desc)
	df_train_fp = data_module.load_and_clean(args.train_fp)
	df_val_desc = data_module.load_and_clean(args.val_desc)
	df_val_fp = data_module.load_and_clean(args.val_fp)

	logger.info(f"Train descriptor shape: {df_train_desc.shape}")
	logger.info(f"Train fingerprint shape: {df_train_fp.shape}")
	logger.info(f"Val descriptor shape: {df_val_desc.shape}")
	logger.info(f"Val fingerprint shape: {df_val_fp.shape}")

	# Extract target and features
	y_train = df_train_desc[args.target]
	X_train_desc = df_train_desc.drop(columns=[args.target])
	X_train_fp = df_train_fp.drop(columns=[args.target])

	y_val = df_val_desc[args.target]
	X_val_desc = df_val_desc.drop(columns=[args.target])
	X_val_fp = df_val_fp.drop(columns=[args.target])

	# --- Feature engineering ---
	logger.info("Applying feature engineering...")
	X_train_desc = features.engineer_features(X_train_desc)
	X_train_fp = features.engineer_features(X_train_fp)
	X_val_desc = features.engineer_features(X_val_desc)
	X_val_fp = features.engineer_features(X_val_fp)

	logger.info(f"Descriptor features after engineering: {X_train_desc.shape[1]}")
	logger.info(f"Fingerprint features after engineering: {X_train_fp.shape[1]}")

	# --- Train meta-ensemble ---
	logger.info("Training meta-ensemble with out-of-fold CV...")
	ensemble = MetaEnsemble(n_folds=5)

	def train_desc_model(X, y):
		"""Train CatBoost on descriptor features."""
		return models.train_model(X, y, model_type='catboost', random_state=args.seed)

	def train_fp_model(X, y):
		"""Train XGBoost on fingerprint features."""
		return models.train_model(X, y, model_type='xgb', random_state=args.seed)

	ensemble.fit(X_train_desc, X_train_fp, y_train, train_desc_model, train_fp_model)

	# --- Evaluate on validation set ---
	logger.info("Evaluating meta-ensemble on validation set...")
	y_pred, y_prob = ensemble.predict(X_val_desc, X_val_fp)
	metrics = evaluate.evaluate_classification(y_val, y_pred, y_prob)

	# --- Save ensemble ---
	utils.ensure_dir(os.path.dirname(os.path.abspath(args.output_ensemble)))
	ensemble.save(args.output_ensemble)
	logger.info(f"Meta-ensemble saved to: {args.output_ensemble}")

	# --- Save metrics ---
	utils.ensure_dir(os.path.dirname(os.path.abspath(args.output_metrics)))
	with open(args.output_metrics, 'w') as f:
		f.write("=== Meta-Ensemble Evaluation Metrics ===\n\n")
		for k, v in metrics.items():
			if k not in ('confusion_matrix', 'report'):
				f.write(f"{k}: {v}\n")
		f.write(f"\nConfusion matrix:\n{metrics['confusion_matrix']}\n")
		f.write(f"\nClassification report:\n{metrics['report']}\n")
		f.write(f"\nMeta-Ensemble Configuration:\n")
		f.write(f"  train_desc: {args.train_desc}\n")
		f.write(f"  train_fp: {args.train_fp}\n")
		f.write(f"  val_desc: {args.val_desc}\n")
		f.write(f"  val_fp: {args.val_fp}\n")
		f.write(f"  base_models: CatBoost (descriptors) + XGBoost (fingerprints)\n")
		f.write(f"  meta_learner: LogisticRegression\n")
		f.write(f"  cv_folds: 5\n")
		f.write(f"  random_seed: {args.seed}\n")

	logger.info(f"Metrics saved to: {args.output_metrics}")
	logger.info(f"AUC: {metrics.get('auc', 'N/A'):.4f}")
	logger.info(f"MCC: {metrics.get('mcc', 'N/A'):.4f}")
	logger.info("Meta-ensemble training complete.")


if __name__ == '__main__':
	main()
