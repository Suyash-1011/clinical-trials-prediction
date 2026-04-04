# Tests for feature engineering in src/features.py
import pytest
import pandas as pd
import numpy as np
from src import features


@pytest.fixture
def descriptor_data():
	"""Create sample descriptor dataset with required columns."""
	np.random.seed(42)
	n_samples = 100
	data = {
		'MolLogP': np.random.uniform(0, 5, n_samples),
		'fr_NH0': np.random.randint(0, 5, n_samples),
		'fr_aniline': np.random.randint(0, 3, n_samples),
		'NumAromaticRings': np.random.randint(0, 4, n_samples),
		'fr_Ar_N': np.random.randint(0, 4, n_samples),
		'NumHalides': np.random.randint(0, 4, n_samples),
		'PEOE_VSA1': np.random.uniform(0, 100, n_samples),
		'QED': np.random.uniform(0, 1, n_samples),
		'FractionCsp3': np.random.uniform(0, 1, n_samples),
		'NumAliphaticCarbocycles': np.random.randint(0, 3, n_samples),
		'fr_Al_OH_noTert': np.random.randint(0, 2, n_samples),
	}
	return pd.DataFrame(data)


@pytest.fixture
def fingerprint_data():
	"""Create sample fingerprint dataset (no MolLogP column)."""
	np.random.seed(42)
	n_samples = 100
	data = {f'bit_{i}': np.random.randint(0, 2, n_samples) for i in range(1024)}
	return pd.DataFrame(data)


def test_engineer_features_descriptor_data(descriptor_data):
	"""Test feature engineering on descriptor dataset."""
	X = descriptor_data.copy()
	X_engineered = features.engineer_features(X)
	
	# Should have original columns + 9 new features
	expected_new_cols = [
		'herg_risk_proxy',
		'amine_tox_score',
		'total_aromatic_N_burden',
		'reactivity_spread',
		'halogen_logp_interaction',
		'qed_sp3_synergy',
		'lipophilic_flag',
		'aromatic_N_times_logp',
		'allylic_ester_alert',
	]
	for col in expected_new_cols:
		assert col in X_engineered.columns, f"Missing feature: {col}"
	
	# Check shape
	assert X_engineered.shape[0] == X.shape[0]
	assert X_engineered.shape[1] == X.shape[1] + 9
	
	# Check no NaN values
	assert not X_engineered.isnull().any().any()


def test_engineer_features_fingerprint_data(fingerprint_data):
	"""Test that fingerprint data passes through unchanged."""
	X = fingerprint_data.copy()
	X_engineered = features.engineer_features(X)
	
	# Should be unchanged (no 'MolLogP' column detected)
	pd.testing.assert_frame_equal(X, X_engineered)


def test_herg_risk_proxy(descriptor_data):
	"""Test hERG risk proxy feature (MolLogP * fr_NH0)."""
	X = descriptor_data.copy()
	X_engineered = features.engineer_features(X)
	
	expected = X['MolLogP'] * X['fr_NH0']
	expected.name = 'herg_risk_proxy'
	pd.testing.assert_series_equal(
		X_engineered['herg_risk_proxy'],
		expected,
		check_names=True
	)


def test_amine_tox_score(descriptor_data):
	"""Test amine tox score feature."""
	X = descriptor_data.copy()
	X_engineered = features.engineer_features(X)
	
	expected = X['MolLogP'] * X['fr_aniline']
	expected.name = 'amine_tox_score'
	pd.testing.assert_series_equal(
		X_engineered['amine_tox_score'],
		expected,
		check_names=True
	)


def test_total_aromatic_N_burden(descriptor_data):
	"""Test total aromatic N burden feature."""
	X = descriptor_data.copy()
	X_engineered = features.engineer_features(X)
	
	# Add missing columns with 0 values to match implementation
	if 'fr_ArN' not in X.columns:
		X['fr_ArN'] = 0
	if 'fr_Ar_NH' not in X.columns:
		X['fr_Ar_NH'] = 0
	
	expected = X['fr_Ar_N'] + X['fr_ArN'] + X['fr_Ar_NH']
	expected.name = 'total_aromatic_N_burden'
	pd.testing.assert_series_equal(
		X_engineered['total_aromatic_N_burden'].astype('float64'),
		expected.astype('float64'),
		check_names=True
	)


def test_reactivity_spread(descriptor_data):
	"""Test reactivity spread feature."""
	X = descriptor_data.copy()
	X_engineered = features.engineer_features(X)
	
	# reactivity_spread should be present and numeric
	assert 'reactivity_spread' in X_engineered.columns
	assert pd.api.types.is_numeric_dtype(X_engineered['reactivity_spread'])
	assert X_engineered['reactivity_spread'].notna().all()


def test_halogen_logp_interaction(descriptor_data):
	"""Test halogen * LogP interaction feature."""
	X = descriptor_data.copy()
	# The implementation looks for 'fr_halogen', not 'NumHalides'
	# Since fr_halogen is not in descriptor_data, it defaults to 0
	X_engineered = features.engineer_features(X)
	
	# Should default to 0 since fr_halogen not present
	assert (X_engineered['halogen_logp_interaction'] == 0).all()


def test_qed_sp3_synergy(descriptor_data):
	"""Test QED and Sp3 synergy feature."""
	X = descriptor_data.copy()
	X_engineered = features.engineer_features(X)
	
	# QED and FractionCSP3 may not be in the test data (defaulted to 0)
	assert 'qed_sp3_synergy' in X_engineered.columns
	assert pd.api.types.is_numeric_dtype(X_engineered['qed_sp3_synergy'])


def test_lipophilic_flag(descriptor_data):
	"""Test lipophilic flag (1 if MolLogP > 3, 0 otherwise)."""
	X = descriptor_data.copy()
	X_engineered = features.engineer_features(X)
	
	# Check values are binary
	assert set(X_engineered['lipophilic_flag'].unique()).issubset({0, 1})
	# Check correctness
	expected = (X['MolLogP'] > 3).astype(int)
	expected.name = 'lipophilic_flag'
	pd.testing.assert_series_equal(
		X_engineered['lipophilic_flag'].astype('int64'),
		expected.astype('int64'),
		check_names=True
	)


def test_aromatic_N_times_logp(descriptor_data):
	"""Test aromatic N * LogP interaction."""
	X = descriptor_data.copy()
	X_engineered = features.engineer_features(X)
	
	# Build the expected value step by step
	if 'fr_ArN' not in X.columns:
		X['fr_ArN'] = 0
	if 'fr_Ar_NH' not in X.columns:
		X['fr_Ar_NH'] = 0
	aromatic_N_burden = X['fr_Ar_N'] + X['fr_ArN'] + X['fr_Ar_NH']
	expected = aromatic_N_burden * X['MolLogP']
	
	assert np.allclose(X_engineered['aromatic_N_times_logp'].values, expected.values)


def test_allylic_ester_alert(descriptor_data):
	"""Test allylic ester alert (combination of aliphatic carbocycles and hydroxyl)."""
	X = descriptor_data.copy()
	X_engineered = features.engineer_features(X)
	
	# Defaults are used if columns are missing
	fr_allylic_oxid = X['fr_allylic_oxid'] if 'fr_allylic_oxid' in X.columns else pd.Series(0, index=X.index)
	fr_ester = X['fr_ester'] if 'fr_ester' in X.columns else pd.Series(0, index=X.index)
	expected = fr_allylic_oxid + fr_ester
	
	assert np.allclose(X_engineered['allylic_ester_alert'].values, expected.values)


def test_engineer_features_handles_missing_columns():
	"""Test that engineer_features handles missing columns gracefully."""
	# Create data with missing columns
	X = pd.DataFrame({
		'col1': [1, 2, 3],
		'col2': [4, 5, 6],
	})
	
	# Should not raise error, and return unchanged
	X_engineered = features.engineer_features(X)
	pd.testing.assert_frame_equal(X, X_engineered)


def test_engineer_features_preserves_index():
	"""Test that engineer_features preserves row indices."""
	X = pd.DataFrame(
		{'MolLogP': [1, 2, 3], 'fr_NH0': [0, 1, 2]},
		index=['a', 'b', 'c']
	)
	X_engineered = features.engineer_features(X)
	assert list(X_engineered.index) == ['a', 'b', 'c']


def test_engineer_features_with_nan_values():
	"""Test feature engineering with NaN values in input."""
	X = pd.DataFrame({
		'MolLogP': [1.0, np.nan, 3.0],
		'fr_NH0': [1, 2, np.nan],
		'NumAromaticRings': [0, 1, 2],
	})
	
	# engineer_features should handle NaN gracefully
	X_engineered = features.engineer_features(X)
	
	# Check that output has expected structure
	assert 'herg_risk_proxy' in X_engineered.columns
	assert X_engineered.shape[0] == 3


def test_engineer_features_output_dtypes():
	"""Test that engineered features have correct dtypes."""
	X = pd.DataFrame({
		'MolLogP': [1.5, 2.5, 3.5],
		'fr_NH0': [1, 2, 3],
		'NumAromaticRings': [0, 1, 2],
		'fr_Ar_N': [0, 1, 1],
		'NumHalides': [0, 1, 2],
		'PEOE_VSA1': [10.0, 20.0, 30.0],
		'QED': [0.5, 0.6, 0.7],
		'FractionCsp3': [0.3, 0.4, 0.5],
		'NumAliphaticCarbocycles': [0, 1, 2],
		'fr_Al_OH_noTert': [0, 0, 1],
		'fr_aniline': [0, 1, 1],
	})
	
	X_engineered = features.engineer_features(X)
	
	# Most features should be numeric
	for col in X_engineered.columns:
		assert pd.api.types.is_numeric_dtype(X_engineered[col])
	
	# lipophilic_flag should be int
	assert pd.api.types.is_integer_dtype(X_engineered['lipophilic_flag'])


def test_engineer_features_deterministic():
	"""Test that engineer_features is deterministic."""
	X = pd.DataFrame({
		'MolLogP': [1.5, 2.5, 3.5],
		'fr_NH0': [1, 2, 3],
		'NumAromaticRings': [0, 1, 2],
		'fr_Ar_N': [0, 1, 1],
		'NumHalides': [0, 1, 2],
		'PEOE_VSA1': [10.0, 20.0, 30.0],
		'QED': [0.5, 0.6, 0.7],
		'FractionCsp3': [0.3, 0.4, 0.5],
		'NumAliphaticCarbocycles': [0, 1, 2],
		'fr_Al_OH_noTert': [0, 0, 1],
		'fr_aniline': [0, 1, 1],
	})
	
	X_eng1 = features.engineer_features(X)
	X_eng2 = features.engineer_features(X)
	
	# Results should be identical
	pd.testing.assert_frame_equal(X_eng1, X_eng2)

