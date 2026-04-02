
import os
import random
import numpy as np
import logging

def set_seed(seed=42):
	"""
	Set random seed for reproducibility.
	Args:
		seed (int): Random seed.
	"""
	random.seed(seed)
	np.random.seed(seed)
	try:
		import torch
		torch.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
	except ImportError:
		pass

def ensure_dir(path):
	"""
	Ensure a directory exists; create if not.
	Args:
		path (str): Directory path.
	"""
	os.makedirs(path, exist_ok=True)

def get_logger(name=__name__, level=logging.INFO):
	"""
	Get a configured logger.
	Args:
		name (str): Logger name.
		level (int): Logging level.
	Returns:
		logging.Logger: Configured logger.
	"""
	logger = logging.getLogger(name)
	if not logger.handlers:
		handler = logging.StreamHandler()
		formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
		handler.setFormatter(formatter)
		logger.addHandler(handler)
	logger.setLevel(level)
	return logger
