"""
MetaEnsemble: Two-level stacking for combining descriptor and fingerprint models.

This module implements a meta-ensemble that:
1. Trains separate base models on descriptor and fingerprint feature spaces
2. Generates out-of-fold meta-features from both base models
3. Trains a meta-learner (logistic regression) on these meta-features
4. Combines predictions from both feature spaces at inference time

Architecture:
  - Base learners: CatBoost (descriptors) and XGBoost (fingerprints)
  - Meta-learner: Logistic Regression
  - Training: Out-of-fold CV with 5 folds to prevent data leakage
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import logging


logger = logging.getLogger(__name__)


class MetaEnsemble:
	"""
	Two-level stacking ensemble combining descriptor and fingerprint models.
	
	Attributes:
		desc_model: Trained model on descriptor features
		fp_model: Trained model on fingerprint features
		meta_model: Trained logistic regression on stacked predictions
		desc_scaler: StandardScaler for descriptor features
		fp_scaler: StandardScaler for fingerprint features
		desc_cols: Columns to keep for descriptor data
		fp_cols: Columns to keep for fingerprint data
		n_folds: Number of folds for out-of-fold CV (default 5)
	"""
	
	def __init__(self, n_folds=5):
		"""Initialize meta-ensemble with number of CV folds."""
		self.desc_model = None
		self.fp_model = None
		self.meta_model = None
		self.desc_scaler = None
		self.fp_scaler = None
		self.desc_cols = None
		self.fp_cols = None
		self.n_folds = n_folds
		logger.info(f"Initialized MetaEnsemble with {n_folds} folds")
	
	def fit(self, X_desc, X_fp, y, desc_model_func, fp_model_func):
		"""
		Train meta-ensemble on descriptor and fingerprint data.
		
		Args:
			X_desc: Descriptor features (DataFrame)
			X_fp: Fingerprint features (DataFrame)
			y: Target labels (Series or array)
			desc_model_func: Function to train descriptor model (e.g., models.train_model with model_type='catboost')
			fp_model_func: Function to train fingerprint model (e.g., models.train_model with model_type='xgb')
		
		Returns:
			self
		"""
		n_samples = len(y)
		n_desc_features = X_desc.shape[1]
		n_fp_features = X_fp.shape[1]
		
		logger.info(f"Training MetaEnsemble on {n_samples} samples")
		logger.info(f"  Descriptors: {n_desc_features} features")
		logger.info(f"  Fingerprints: {n_fp_features} features")
		
		# Initialize out-of-fold meta-features
		meta_features = np.zeros((n_samples, 2))  # 2 columns: desc_prob, fp_prob
		
		# Out-of-fold CV
		cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
		fold_idx = 0
		
		for train_idx, val_idx in cv.split(X_desc, y):
			fold_idx += 1
			logger.info(f"  Fold {fold_idx}/{self.n_folds}")
			
			# Split descriptor data
			X_desc_train = X_desc.iloc[train_idx].copy()
			X_desc_val = X_desc.iloc[val_idx].copy()
			y_train = y.iloc[train_idx] if isinstance(y, pd.Series) else y[train_idx]
			y_val = y.iloc[val_idx] if isinstance(y, pd.Series) else y[val_idx]
			
			# Split fingerprint data (same folds as descriptors)
			X_fp_train = X_fp.iloc[train_idx].copy()
			X_fp_val = X_fp.iloc[val_idx].copy()
			
			# Scale descriptor data
			desc_scaler = StandardScaler()
			X_desc_train_scaled = desc_scaler.fit_transform(X_desc_train)
			X_desc_train_scaled = pd.DataFrame(X_desc_train_scaled, columns=X_desc_train.columns, index=X_desc_train.index)
			X_desc_val_scaled = desc_scaler.transform(X_desc_val)
			X_desc_val_scaled = pd.DataFrame(X_desc_val_scaled, columns=X_desc_val.columns, index=X_desc_val.index)
			
			# Scale fingerprint data
			fp_scaler = StandardScaler()
			X_fp_train_scaled = fp_scaler.fit_transform(X_fp_train)
			X_fp_train_scaled = pd.DataFrame(X_fp_train_scaled, columns=X_fp_train.columns, index=X_fp_train.index)
			X_fp_val_scaled = fp_scaler.transform(X_fp_val)
			X_fp_val_scaled = pd.DataFrame(X_fp_val_scaled, columns=X_fp_val.columns, index=X_fp_val.index)
			
			# Train descriptor model on this fold
			desc_model = desc_model_func(X_desc_train_scaled, y_train)
			_, desc_probs = self._predict_proba(desc_model, X_desc_val_scaled)
			meta_features[val_idx, 0] = desc_probs if desc_probs.ndim == 1 else desc_probs[:, 1]
			
			# Train fingerprint model on this fold
			fp_model = fp_model_func(X_fp_train_scaled, y_train)
			_, fp_probs = self._predict_proba(fp_model, X_fp_val_scaled)
			meta_features[val_idx, 1] = fp_probs if fp_probs.ndim == 1 else fp_probs[:, 1]
		
		logger.info("Out-of-fold meta-features generated")
		
		# Train full models on all data for final ensemble
		logger.info("Training full base models on all data...")
		
		# Descriptor model
		desc_scaler_full = StandardScaler()
		X_desc_scaled_full = desc_scaler_full.fit_transform(X_desc)
		X_desc_scaled_full = pd.DataFrame(X_desc_scaled_full, columns=X_desc.columns)
		self.desc_model = desc_model_func(X_desc_scaled_full, y)
		self.desc_scaler = desc_scaler_full
		self.desc_cols = list(X_desc.columns)
		
		# Fingerprint model
		fp_scaler_full = StandardScaler()
		X_fp_scaled_full = fp_scaler_full.fit_transform(X_fp)
		X_fp_scaled_full = pd.DataFrame(X_fp_scaled_full, columns=X_fp.columns)
		self.fp_model = fp_model_func(X_fp_scaled_full, y)
		self.fp_scaler = fp_scaler_full
		self.fp_cols = list(X_fp.columns)
		
		# Train meta-learner
		logger.info("Training meta-learner (LogisticRegression)...")
		self.meta_model = LogisticRegression(max_iter=1000, random_state=42)
		self.meta_model.fit(meta_features, y)
		
		logger.info("MetaEnsemble training complete")
		return self
	
	def _predict_proba(self, model, X):
		"""
		Predict probabilities, handling different model types.
		
		Args:
			model: Trained model (sklearn, XGBoost, or CatBoost)
			X: Feature matrix
		
		Returns:
			y_pred: Class predictions (0 or 1)
			y_prob: Probability of positive class
		"""
		# Try sklearn-style predict_proba first
		if hasattr(model, 'predict_proba'):
			y_probs = model.predict_proba(X)
			y_pred = (y_probs[:, 1] > 0.5).astype(int)
			return y_pred, y_probs[:, 1]
		# Then try XGBoost/CatBoost style
		elif hasattr(model, 'predict'):
			y_probs = model.predict(X)
			# If probabilities are returned, use them; otherwise assume they're predictions
			if y_probs.ndim > 1 and y_probs.shape[1] == 2:
				y_pred = np.argmax(y_probs, axis=1)
				return y_pred, y_probs[:, 1]
			else:
				y_pred = (y_probs > 0.5).astype(int)
				return y_pred, y_probs
		else:
			raise ValueError(f"Model {type(model)} has no predict_proba or predict method")
	
	def predict(self, X_desc, X_fp):
		"""
		Make predictions using the meta-ensemble.
		
		Args:
			X_desc: Descriptor features (DataFrame)
			X_fp: Fingerprint features (DataFrame)
		
		Returns:
			y_pred: Class predictions (0 or 1)
			y_prob: Probability of positive class
		"""
		if self.desc_model is None or self.fp_model is None or self.meta_model is None:
			raise ValueError("MetaEnsemble not fitted yet. Call fit() first.")
		
		# Scale descriptor data
		X_desc_scaled = self.desc_scaler.transform(X_desc)
		X_desc_scaled = pd.DataFrame(X_desc_scaled, columns=X_desc.columns, index=X_desc.index)
		
		# Scale fingerprint data
		X_fp_scaled = self.fp_scaler.transform(X_fp)
		X_fp_scaled = pd.DataFrame(X_fp_scaled, columns=X_fp.columns, index=X_fp.index)
		
		# Get predictions from base models
		_, desc_probs = self._predict_proba(self.desc_model, X_desc_scaled[self.desc_cols])
		_, fp_probs = self._predict_proba(self.fp_model, X_fp_scaled[self.fp_cols])
		
		# Stack meta-features
		meta_features = np.column_stack([desc_probs, fp_probs])
		
		# Predict with meta-learner
		y_prob = self.meta_model.predict_proba(meta_features)[:, 1]
		y_pred = self.meta_model.predict(meta_features)
		
		return y_pred, y_prob
	
	def save(self, path):
		"""
		Save meta-ensemble to disk.
		
		Args:
			path: Path to save the ensemble bundle
		"""
		bundle = {
			'desc_model': self.desc_model,
			'fp_model': self.fp_model,
			'meta_model': self.meta_model,
			'desc_scaler': self.desc_scaler,
			'fp_scaler': self.fp_scaler,
			'desc_cols': self.desc_cols,
			'fp_cols': self.fp_cols,
			'n_folds': self.n_folds,
		}
		joblib.dump(bundle, path)
		logger.info(f"MetaEnsemble saved to {path}")
	
	@classmethod
	def load(cls, path):
		"""
		Load meta-ensemble from disk.
		
		Args:
			path: Path to saved ensemble bundle
		
		Returns:
			MetaEnsemble instance with loaded weights
		"""
		bundle = joblib.load(path)
		ensemble = cls(n_folds=bundle['n_folds'])
		ensemble.desc_model = bundle['desc_model']
		ensemble.fp_model = bundle['fp_model']
		ensemble.meta_model = bundle['meta_model']
		ensemble.desc_scaler = bundle['desc_scaler']
		ensemble.fp_scaler = bundle['fp_scaler']
		ensemble.desc_cols = bundle['desc_cols']
		ensemble.fp_cols = bundle['fp_cols']
		logger.info(f"MetaEnsemble loaded from {path}")
		return ensemble
