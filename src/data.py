# Data loading, cleaning, and validation utilities
import pandas as pd
import os

def load_data(file_path):
	"""
	Load a CSV file into a pandas DataFrame.
	Args:
		file_path (str): Path to the CSV file.
	Returns:
		pd.DataFrame: Loaded data.
	"""
	if not os.path.exists(file_path):
		raise FileNotFoundError(f"File not found: {file_path}")
	return pd.read_csv(file_path)

def clean_data(df):
	"""
	Basic cleaning: drop duplicates, reset index, handle missing values (simple strategy).
	Args:
		df (pd.DataFrame): Input DataFrame.
	Returns:
		pd.DataFrame: Cleaned DataFrame.
	"""
	df = df.drop_duplicates().reset_index(drop=True)
	# Fill missing values with column mean (for numeric columns)
	for col in df.select_dtypes(include=['float', 'int']).columns:
		df[col] = df[col].fillna(df[col].mean())
	# For categorical columns, fill with mode
	for col in df.select_dtypes(include=['object']).columns:
		df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else '')
	return df

def load_and_clean(file_path):
	"""
	Load and clean a CSV file.
	Args:
		file_path (str): Path to the CSV file.
	Returns:
		pd.DataFrame: Cleaned DataFrame.
	"""
	df = load_data(file_path)
	return clean_data(df)
