# Tests for advanced features: ensembling, tuning, SHAP
import pytest
import numpy as np
import pandas as pd
from src import models

def sample_data():
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(40, 4), columns=[f'f{i}' for i in range(4)])
    y = pd.Series(np.random.randint(0, 2, 40))
    return X, y

def test_build_ensemble_voting():
    X, y = sample_data()
    # build_ensemble returns a fitted model (VotingClassifier or StackingClassifier)
    model = models.build_ensemble(X, y, method='voting')
    y_pred, y_prob = models.predict(model, X)
    assert len(y_pred) == len(y)
    # Voting classifier may or may not return probabilities depending on base models
    if y_prob is not None:
        assert y_prob.shape[0] == len(y)

def test_build_ensemble_stacking():
    X, y = sample_data()
    model = models.build_ensemble(X, y, method='stacking')
    y_pred, y_prob = models.predict(model, X)
    assert len(y_pred) == len(y)
    # Stacking with LogisticRegression meta-model should return probabilities
    if y_prob is not None:
        assert y_prob.shape[0] == len(y)

def test_tune_hyperparameters_runs():
    X, y = sample_data()
    # Tune with just 2 trials for speed
    params = models.tune_hyperparameters(X, y, model_type='rf', n_trials=2, timeout=60)
    assert isinstance(params, dict)
    # Should return some hyperparameters for random forest
    assert len(params) > 0
