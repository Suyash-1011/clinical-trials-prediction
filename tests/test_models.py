# Tests for model training and prediction utilities in src/models.py
import pytest
import pandas as pd
import numpy as np
from src import models

def sample_data():
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(100, 5), columns=[f'f{i}' for i in range(5)])
    y = pd.Series(np.random.randint(0, 2, 100))
    return X, y

@pytest.mark.parametrize("model_type", ['rf', 'svm', 'xgb', 'catboost', 'logreg'])
def test_train_and_predict(model_type):
    X, y = sample_data()
    model = models.train_model(X, y, model_type=model_type)
    y_pred, y_prob = models.predict(model, X)
    assert len(y_pred) == len(y)
    if y_prob is not None:
        assert y_prob.shape[0] == len(y)

    # Test save/load
    models.save_model(model, f"/tmp/test_{model_type}.joblib")
    loaded = models.load_model(f"/tmp/test_{model_type}.joblib")
    y_pred2, y_prob2 = models.predict(loaded, X)
    np.testing.assert_array_equal(y_pred, y_pred2)
    if y_prob is not None and y_prob2 is not None:
        np.testing.assert_allclose(y_prob, y_prob2, rtol=1e-5, atol=1e-5)
