
import numpy as np
import pandas as pd
from sklearn.metrics import (
	accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
	confusion_matrix, classification_report, matthews_corrcoef, brier_score_loss, cohen_kappa_score
)
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_classification(y_true, y_pred, y_prob=None):
	"""
	Compute standard classification metrics.
	Args:
		y_true (array-like): True labels.
		y_pred (array-like): Predicted labels.
		y_prob (array-like, optional): Predicted probabilities for class 1.
	Returns:
		dict: Metrics (accuracy, precision, recall, f1, auc, mcc, kappa, brier, confusion_matrix, report)
	"""
	metrics = {
		'accuracy': accuracy_score(y_true, y_pred),
		'precision': precision_score(y_true, y_pred, zero_division=0),
		'recall': recall_score(y_true, y_pred, zero_division=0),
		'f1': f1_score(y_true, y_pred, zero_division=0),
		'mcc': matthews_corrcoef(y_true, y_pred),
		'kappa': cohen_kappa_score(y_true, y_pred),
		'confusion_matrix': confusion_matrix(y_true, y_pred),
		'report': classification_report(y_true, y_pred, zero_division=0, output_dict=True)
	}
	if y_prob is not None:
		metrics['auc'] = roc_auc_score(y_true, y_prob)
		metrics['brier'] = brier_score_loss(y_true, y_prob)
	else:
		metrics['auc'] = None
		metrics['brier'] = None
	return metrics

def plot_confusion_matrix(cm, class_names=None, figsize=(6, 5), title='Confusion Matrix'):
	"""
	Plot a confusion matrix using seaborn heatmap.
	Args:
		cm (np.ndarray): Confusion matrix.
		class_names (list): Class names.
		figsize (tuple): Figure size.
		title (str): Plot title.
	Returns:
		matplotlib.figure.Figure: The figure object.
	"""
	fig, ax = plt.subplots(figsize=figsize)
	sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
				xticklabels=class_names, yticklabels=class_names, ax=ax, annot_kws={"size": 16})
	ax.set_xlabel('Predicted')
	ax.set_ylabel('True')
	ax.set_title(title)
	plt.tight_layout()
	return fig
