# Tests for evaluation and reporting utilities in src/evaluate.py
import pytest
import numpy as np
from src import evaluate

def test_evaluate_classification_basic():
    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 0, 1, 1, 1, 0])
    y_prob = np.array([0.1, 0.8, 0.4, 0.2, 0.7, 0.6, 0.9, 0.3])
    metrics = evaluate.evaluate_classification(y_true, y_pred, y_prob)
    assert 'accuracy' in metrics and 0 <= metrics['accuracy'] <= 1
    assert 'auc' in metrics and 0 <= metrics['auc'] <= 1
    assert 'confusion_matrix' in metrics
    assert 'report' in metrics

def test_plot_confusion_matrix_runs():
    cm = np.array([[5, 2], [1, 7]])
    fig = evaluate.plot_confusion_matrix(cm, class_names=['Class 0', 'Class 1'])
    assert fig is not None
