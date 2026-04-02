def build_ensemble(X, y, method='voting', base_models=None, meta_model=None):
	"""
	Build an ensemble model (Voting or Stacking).
	Args:
		X (pd.DataFrame): Feature matrix.
		y (pd.Series): Target vector.
		method (str): 'voting' or 'stacking'.
		base_models (list): List of (str, estimator) tuples for stacking (optional).
		meta_model: Meta-model for stacking (optional).
	Returns:
		Trained ensemble model.
	"""
	if method == 'voting':
		estimators = [
			('rf', RandomForestClassifier(n_jobs=-1, random_state=42)),
			('svm', SVC(probability=True, random_state=42)),
			('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_jobs=-1, random_state=42)),
			('cat', CatBoostClassifier(verbose=0, random_state=42)),
		]
		ensemble = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
		ensemble.fit(X, y)
		return ensemble
	elif method == 'stacking':
		if base_models is None:
			base_models = [
				('rf', RandomForestClassifier(n_jobs=-1, random_state=42)),
				('svm', SVC(probability=True, random_state=42)),
				('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_jobs=-1, random_state=42)),
				('cat', CatBoostClassifier(verbose=0, random_state=42)),
			]
		if meta_model is None:
			meta_model = LogisticRegression(max_iter=1000, random_state=42)
		ensemble = StackingClassifier(estimators=base_models, final_estimator=meta_model, n_jobs=-1, passthrough=False)
		ensemble.fit(X, y)
		return ensemble
	else:
		raise ValueError("Unknown ensemble method: choose 'voting' or 'stacking'")


import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
import optuna
import shap

def train_model(X, y, model_type='rf', **kwargs):
	"""
	Train a model of the specified type.

import optuna
import shap

	Args:
		X (pd.DataFrame): Feature matrix.
		y (pd.Series): Target vector.
		model_type (str): 'rf', 'svm', 'xgb', 'catboost', 'logreg'.
		**kwargs: Model hyperparameters.
	Returns:
		Trained model instance.
	"""
	if model_type == 'rf':
		model = RandomForestClassifier(n_jobs=-1, **kwargs)
	elif model_type == 'svm':
		model = SVC(probability=True, **kwargs)
	elif model_type == 'xgb':
		model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_jobs=-1, **kwargs)
	elif model_type == 'catboost':
		model = CatBoostClassifier(verbose=0, **kwargs)
	elif model_type == 'logreg':
		model = LogisticRegression(max_iter=1000, **kwargs)
	else:
		raise ValueError(f"Unknown model_type: {model_type}")
	model.fit(X, y)
	return model

def predict(model, X):
	"""
	Predict class labels and probabilities.
	Args:
		model: Trained model.
		X (pd.DataFrame): Feature matrix.
	Returns:
		np.ndarray: Predicted class labels.
		np.ndarray: Predicted probabilities (for class 1).
	"""
	y_pred = model.predict(X)
	if hasattr(model, 'predict_proba'):
		y_prob = model.predict_proba(X)[:, 1]
	else:
		y_prob = None
	return y_pred, y_prob

def tune_hyperparameters(X, y, model_type='rf', n_trials=20, timeout=600, seed=42):
	"""
	Hyperparameter tuning using Optuna.
	Args:
		X (pd.DataFrame): Feature matrix.
		y (pd.Series): Target vector.
		model_type (str): 'rf', 'xgb', 'catboost', 'svm', 'logreg'.
		n_trials (int): Number of Optuna trials.
		timeout (int): Timeout in seconds.
		seed (int): Random seed.
	Returns:
		dict: Best parameters.
	"""
	def objective(trial):
		if model_type == 'rf':
			params = {
				'n_estimators': trial.suggest_int('n_estimators', 50, 300),
				'max_depth': trial.suggest_int('max_depth', 3, 20),
				'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
				'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
			}
			model = RandomForestClassifier(**params, n_jobs=-1, random_state=seed)
		elif model_type == 'xgb':
			params = {
				'n_estimators': trial.suggest_int('n_estimators', 50, 300),
				'max_depth': trial.suggest_int('max_depth', 3, 20),
				'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
				'subsample': trial.suggest_float('subsample', 0.5, 1.0),
			}
			model = XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss', n_jobs=-1, random_state=seed)
		elif model_type == 'catboost':
			params = {
				'iterations': trial.suggest_int('iterations', 50, 300),
				'depth': trial.suggest_int('depth', 3, 10),
				'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
			}
			model = CatBoostClassifier(**params, verbose=0, random_state=seed)
		elif model_type == 'svm':
			params = {
				'C': trial.suggest_float('C', 0.1, 10.0),
				'gamma': trial.suggest_float('gamma', 1e-4, 1e-1, log=True),
			}
			model = SVC(probability=True, **params, random_state=seed)
		elif model_type == 'logreg':
			params = {
				'C': trial.suggest_float('C', 0.01, 10.0),
			}
			model = LogisticRegression(max_iter=1000, **params, random_state=seed)
		else:
			raise ValueError(f"Unknown model_type: {model_type}")
		from sklearn.model_selection import cross_val_score
		score = cross_val_score(model, X, y, cv=3, scoring='roc_auc', n_jobs=-1).mean()
		return score
	study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed))
	study.optimize(objective, n_trials=n_trials, timeout=timeout)
	return study.best_params

def explain_model_shap(model, X, max_display=10):
	"""
	Compute and plot SHAP values for model interpretability.
	Args:
		model: Trained model.
		X (pd.DataFrame): Feature matrix.
		max_display (int): Number of features to display.
	Returns:
		shap.Explanation: SHAP values object.
	"""
	explainer = shap.Explainer(model, X)
	shap_values = explainer(X)
	shap.summary_plot(shap_values, X, max_display=max_display, show=False)
	return shap_values

def save_model(model, file_path):
	"""
	Save a trained model to disk.
	Args:
		model: Trained model.
		file_path (str): Path to save the model.
	"""
	joblib.dump(model, file_path)

def load_model(file_path):
	"""
	Load a trained model from disk.
	Args:
		file_path (str): Path to the saved model.
	Returns:
		Loaded model instance.
	"""
	return joblib.load(file_path)
