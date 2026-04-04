# Integration tests for the end-to-end pipeline CLI
import subprocess
import tempfile
import os
import pandas as pd
import numpy as np
import joblib
import pytest


@pytest.fixture
def synthetic_descriptor_data():
	"""Create synthetic descriptor dataset for testing."""
	np.random.seed(42)
	n_train = 50
	n_val = 20
	
	# Descriptor columns
	cols = [
		'MolLogP', 'fr_NH0', 'fr_aniline', 'NumAromaticRings', 'fr_Ar_N',
		'NumHalides', 'PEOE_VSA1', 'QED', 'FractionCsp3',
		'NumAliphaticCarbocycles', 'fr_Al_OH_noTert'
	]
	
	# Training data
	train_data = {col: np.random.uniform(0, 5, n_train) if col in ['MolLogP', 'PEOE_VSA1', 'QED', 'FractionCsp3']
								  else np.random.randint(0, 4, n_train) for col in cols}
	train_data['Target'] = np.random.randint(0, 2, n_train)
	df_train = pd.DataFrame(train_data)
	
	# Validation data
	val_data = {col: np.random.uniform(0, 5, n_val) if col in ['MolLogP', 'PEOE_VSA1', 'QED', 'FractionCsp3']
								else np.random.randint(0, 4, n_val) for col in cols}
	val_data['Target'] = np.random.randint(0, 2, n_val)
	df_val = pd.DataFrame(val_data)
	
	return df_train, df_val


def test_pipeline_cli_trains_model(synthetic_descriptor_data):
	"""Test that pipeline_cli.py trains a model with the new --train/--val interface."""
	df_train, df_val = synthetic_descriptor_data
	
	with tempfile.TemporaryDirectory() as tmpdir:
		# Write data files
		train_path = os.path.join(tmpdir, 'train.csv')
		val_path = os.path.join(tmpdir, 'val.csv')
		model_path = os.path.join(tmpdir, 'model.joblib')
		metrics_path = os.path.join(tmpdir, 'metrics.txt')
		
		df_train.to_csv(train_path, index=False)
		df_val.to_csv(val_path, index=False)
		
		# Run pipeline
		script_path = os.path.join(os.path.dirname(__file__), '../scripts/pipeline_cli.py')
		script_path = os.path.abspath(script_path)
		
		result = subprocess.run([
			'python3', script_path,
			'--train', train_path,
			'--val', val_path,
			'--target', 'Target',
			'--model', 'rf',
			'--output_model', model_path,
			'--output_metrics', metrics_path,
			'--seed', '42'
		], capture_output=True, text=True)
		
		# Check execution
		assert result.returncode == 0, f"Pipeline failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
		assert os.path.exists(model_path), "Model file not created"
		assert os.path.exists(metrics_path), "Metrics file not created"
		
		# Check model bundle structure
		bundle = joblib.load(model_path)
		assert 'model' in bundle, "Bundle missing 'model' key"
		assert 'scaler' in bundle, "Bundle missing 'scaler' key"
		assert 'cols_to_keep' in bundle, "Bundle missing 'cols_to_keep' key"
		assert 'feature_names_after_engineering' in bundle, "Bundle missing 'feature_names_after_engineering' key"
		assert hasattr(bundle['scaler'], 'transform'), "Scaler not properly fitted"
		
		# Check metrics file
		with open(metrics_path) as f:
			content = f.read()
			assert 'accuracy' in content, "Metrics missing 'accuracy'"
			assert 'auc' in content, "Metrics missing 'auc'"
			assert 'mcc' in content, "Metrics missing 'mcc'"
			assert 'Confusion matrix' in content, "Metrics missing 'Confusion matrix'"
			assert 'Pipeline settings' in content, "Metrics missing pipeline settings"


def test_pipeline_cli_different_models(synthetic_descriptor_data):
	"""Test pipeline with different model types."""
	df_train, df_val = synthetic_descriptor_data
	
	with tempfile.TemporaryDirectory() as tmpdir:
		train_path = os.path.join(tmpdir, 'train.csv')
		val_path = os.path.join(tmpdir, 'val.csv')
		df_train.to_csv(train_path, index=False)
		df_val.to_csv(val_path, index=False)
		
		script_path = os.path.abspath(
			os.path.join(os.path.dirname(__file__), '../scripts/pipeline_cli.py')
		)
		
		for model_type in ['rf', 'xgb', 'catboost', 'logreg']:
			model_path = os.path.join(tmpdir, f'model_{model_type}.joblib')
			metrics_path = os.path.join(tmpdir, f'metrics_{model_type}.txt')
			
			result = subprocess.run([
				'python3', script_path,
				'--train', train_path,
				'--val', val_path,
				'--target', 'Target',
				'--model', model_type,
				'--output_model', model_path,
				'--output_metrics', metrics_path,
			], capture_output=True, text=True)
			
			assert result.returncode == 0, f"Pipeline with {model_type} failed:\n{result.stderr}"
			assert os.path.exists(model_path), f"Model file for {model_type} not created"


def test_pipeline_cli_custom_correlation_threshold(synthetic_descriptor_data):
	"""Test pipeline with custom correlation threshold."""
	df_train, df_val = synthetic_descriptor_data
	
	with tempfile.TemporaryDirectory() as tmpdir:
		train_path = os.path.join(tmpdir, 'train.csv')
		val_path = os.path.join(tmpdir, 'val.csv')
		model_path = os.path.join(tmpdir, 'model.joblib')
		metrics_path = os.path.join(tmpdir, 'metrics.txt')
		
		df_train.to_csv(train_path, index=False)
		df_val.to_csv(val_path, index=False)
		
		script_path = os.path.abspath(
			os.path.join(os.path.dirname(__file__), '../scripts/pipeline_cli.py')
		)
		
		result = subprocess.run([
			'python3', script_path,
			'--train', train_path,
			'--val', val_path,
			'--target', 'Target',
			'--model', 'rf',
			'--output_model', model_path,
			'--output_metrics', metrics_path,
			'--corr_threshold', '0.95',
		], capture_output=True, text=True)
		
		assert result.returncode == 0, f"Pipeline with custom threshold failed:\n{result.stderr}"
		assert os.path.exists(model_path)


def test_predict_cli_uses_model_bundle(synthetic_descriptor_data):
	"""Test that predict_cli.py correctly loads and uses model bundles."""
	df_train, df_val = synthetic_descriptor_data
	
	with tempfile.TemporaryDirectory() as tmpdir:
		# Train a model first
		train_path = os.path.join(tmpdir, 'train.csv')
		val_path = os.path.join(tmpdir, 'val.csv')
		model_path = os.path.join(tmpdir, 'model.joblib')
		metrics_path = os.path.join(tmpdir, 'metrics.txt')
		
		df_train.to_csv(train_path, index=False)
		df_val.to_csv(val_path, index=False)
		
		pipeline_script = os.path.abspath(
			os.path.join(os.path.dirname(__file__), '../scripts/pipeline_cli.py')
		)
		
		result = subprocess.run([
			'python3', pipeline_script,
			'--train', train_path,
			'--val', val_path,
			'--target', 'Target',
			'--model', 'rf',
			'--output_model', model_path,
			'--output_metrics', metrics_path,
		], capture_output=True, text=True)
		
		assert result.returncode == 0
		
		# Now test predict_cli
		test_data_path = os.path.join(tmpdir, 'test.csv')
		predictions_path = os.path.join(tmpdir, 'predictions.csv')
		
		# Use validation data as test data
		df_val_no_target = df_val.drop(columns=['Target'])
		df_val_no_target.to_csv(test_data_path, index=False)
		
		predict_script = os.path.abspath(
			os.path.join(os.path.dirname(__file__), '../scripts/predict_cli.py')
		)
		
		result = subprocess.run([
			'python3', predict_script,
			'--input', test_data_path,
			'--model_bundle', model_path,
			'--output', predictions_path,
		], capture_output=True, text=True)
		
		assert result.returncode == 0, f"Predict CLI failed:\n{result.stderr}"
		assert os.path.exists(predictions_path), "Predictions file not created"
		
		# Check predictions structure
		df_predictions = pd.read_csv(predictions_path)
		assert 'prediction' in df_predictions.columns, "Predictions missing 'prediction' column"
		assert 'probability' in df_predictions.columns, "Predictions missing 'probability' column"
		assert len(df_predictions) == len(df_val_no_target)
		assert all(df_predictions['prediction'].isin([0, 1])), "Predictions not in {0, 1}"
		assert all((df_predictions['probability'] >= 0) & (df_predictions['probability'] <= 1)), "Probabilities out of range"


def test_pipeline_feature_engineering_integration(synthetic_descriptor_data):
	"""Test that pipeline correctly applies feature engineering."""
	df_train, df_val = synthetic_descriptor_data
	
	with tempfile.TemporaryDirectory() as tmpdir:
		train_path = os.path.join(tmpdir, 'train.csv')
		val_path = os.path.join(tmpdir, 'val.csv')
		model_path = os.path.join(tmpdir, 'model.joblib')
		metrics_path = os.path.join(tmpdir, 'metrics.txt')
		
		df_train.to_csv(train_path, index=False)
		df_val.to_csv(val_path, index=False)
		
		script_path = os.path.abspath(
			os.path.join(os.path.dirname(__file__), '../scripts/pipeline_cli.py')
		)
		
		result = subprocess.run([
			'python3', script_path,
			'--train', train_path,
			'--val', val_path,
			'--target', 'Target',
			'--model', 'rf',
			'--output_model', model_path,
			'--output_metrics', metrics_path,
		], capture_output=True, text=True)
		
		assert result.returncode == 0
		
		# Check that bundle contains engineered feature names
		bundle = joblib.load(model_path)
		engineered_features = bundle['feature_names_after_engineering']
		
		# Should contain the 9 engineered features
		expected_features = [
			'herg_risk_proxy', 'amine_tox_score', 'total_aromatic_N_burden',
			'reactivity_spread', 'halogen_logp_interaction', 'qed_sp3_synergy',
			'lipophilic_flag', 'aromatic_N_times_logp', 'allylic_ester_alert'
		]
		
		for feature in expected_features:
			assert feature in engineered_features, f"Feature {feature} not in engineered features"

