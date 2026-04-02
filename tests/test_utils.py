# Tests for general utility functions in src/utils.py
import pytest
import os
import shutil
from src import utils

def test_set_seed_runs():
    utils.set_seed(123)
    # Should not raise

def test_ensure_dir_creates(tmp_path):
    test_dir = tmp_path / "testdir"
    utils.ensure_dir(str(test_dir))
    assert test_dir.exists() and test_dir.is_dir()

def test_get_logger_runs():
    logger = utils.get_logger("test_logger")
    logger.info("Test message")
    assert logger.name == "test_logger"
