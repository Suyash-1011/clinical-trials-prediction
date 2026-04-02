# Tests for feature engineering utilities in src/features.py
import pytest
import pandas as pd
import numpy as np
from src import features

def sample_data():
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(100, 10), columns=[f'f{i}' for i in range(10)])
    y = pd.Series(np.random.randint(0, 2, 100))
    return X, y

def test_scale_features():
    X, _ = sample_data()
    for scaler in ['standard', 'minmax', 'robust']:
        X_scaled = features.scale_features(X, scaler_type=scaler)
        assert X_scaled.shape == X.shape
        assert not X_scaled.isnull().any().any()

def test_select_features_rf():
    X, y = sample_data()
    X_sel, selected = features.select_features_rf(X, y, n_features=5)
    assert X_sel.shape[1] == 5
    assert set(selected).issubset(set(X.columns))

def test_remove_correlated_features():
    X, _ = sample_data()
    X['f_dup'] = X['f0']  # Add a perfectly correlated column
    X_reduced, dropped = features.remove_correlated_features(X, threshold=0.95)
    assert 'f_dup' in dropped
    assert 'f_dup' not in X_reduced.columns

def test_pca_features():
    X, _ = sample_data()
    X_pca, pca = features.pca_features(X, n_components=3)
    assert X_pca.shape[1] == 3
    assert hasattr(pca, 'explained_variance_ratio_')
