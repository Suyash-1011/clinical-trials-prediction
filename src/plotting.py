# Plotting utilities for integrating images and graphs into reports
import matplotlib.pyplot as plt
import seaborn as sns
import os

def save_confusion_matrix(cm, class_names, out_path, title='Confusion Matrix'):
    """
    Save a confusion matrix plot to file.
    Args:
        cm (np.ndarray): Confusion matrix.
        class_names (list): Class names.
        out_path (str): Output image file path.
        title (str): Plot title.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=class_names, yticklabels=class_names, ax=ax, annot_kws={"size": 16})
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

# Additional plotting utilities (ROC, feature importance, etc.) can be added here.
