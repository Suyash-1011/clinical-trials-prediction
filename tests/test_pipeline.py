# Integration test for the end-to-end pipeline CLI
import subprocess
import tempfile
import os
import pandas as pd

def test_pipeline_cli_runs():
    # Create a small synthetic dataset
    df = pd.DataFrame({
        'f1': [0.1, 0.2, 0.3, 0.4],
        'f2': [1, 2, 3, 4],
        'Target': [0, 1, 0, 1]
    })
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, 'input.csv')
        model_path = os.path.join(tmpdir, 'model.joblib')
        metrics_path = os.path.join(tmpdir, 'metrics.txt')
        df.to_csv(input_path, index=False)
        script_path = os.path.join(os.path.dirname(__file__), '../scripts/pipeline_cli.py')
        script_path = os.path.abspath(script_path)
        result = subprocess.run([
            'python3', script_path, input_path, 'Target',
            '--model', 'rf', '--output_model', model_path, '--output_metrics', metrics_path
        ], capture_output=True, text=True, cwd=os.path.dirname(script_path))
        assert result.returncode == 0, result.stderr
        assert os.path.exists(model_path)
        assert os.path.exists(metrics_path)
        with open(metrics_path) as f:
            content = f.read()
            assert 'accuracy' in content
