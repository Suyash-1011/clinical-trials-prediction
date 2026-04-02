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
    model = models.build_ensemble(X, y, method='voting')
    y_pred, y_prob = models.predict(model, X)
    assert len(y_pred) == len(y)
    assert y_prob is not None

def test_build_ensemble_stacking():
    X, y = sample_data()
    model = models.build_ensemble(X, y, method='stacking')
    y_pred, y_prob = models.predict(model, X)
    assert len(y_pred) == len(y)
    assert y_prob is not None

def test_tune_hyperparameters_runs():
    X, y = sample_data()
    params = models.tune_hyperparameters(X, y, model_type='rf', n_trials=2, timeout=60)
    assert isinstance(params, dict)
    assert 'n_estimators' in params

@pytest.mark.filterwarnings('ignore:.*shap.summary_plot.*')
def test_explain_model_shap_runs():
    X, y = sample_data()
    model = models.train_model(X, y, model_type='rf')
    shap_values = models.explain_model_shap(model, X, max_display=2)
    assert shap_values is not None
