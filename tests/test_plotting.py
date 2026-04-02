# Tests for plotting utilities in src/plotting.py
import numpy as np
from src import plotting
import os

def test_save_confusion_matrix(tmp_path):
    cm = np.array([[5, 2], [1, 7]])
    out_path = tmp_path / "cm.png"
    plotting.save_confusion_matrix(cm, ['Class 0', 'Class 1'], str(out_path))
    assert os.path.exists(out_path)
