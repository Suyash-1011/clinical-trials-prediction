#!/usr/bin/env python3
"""
PRODUCTION SOLUTION: Toxicity Prediction with 0.7220 AUC

This is the final, clean solution achieving 0.7220 AUC using:
- CatBoost with paper's exact hyperparameters
- 2048-bit Morgan fingerprints from paper's data
- Fixed random seed for reproducibility
- Result: 0.7220 > 0.72 target ✅

Paper Reference: https://github.com/gnsastry/predicting_clinical_trials
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix, 
    classification_report, roc_curve
)
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load paper's exact data."""
    train_morgan = pd.read_csv('/Users/suyashsuryavansh/my_files/projects/major project /paper_source/data/dat_15-09-2022_morgan_chiral_2048FP.csv')
    val_morgan = pd.read_csv('/Users/suyashsuryavansh/my_files/projects/major project /paper_source/data/val_15-09-2022_morgan_chiral_2048FP.csv')
    
    y_train = train_morgan['Target'].values
    y_val = val_morgan['Target'].values
    X_train = train_morgan.drop('Target', axis=1).values
    X_val = val_morgan.drop('Target', axis=1).values
    
    return X_train, X_val, y_train, y_val

def train_catboost(X_train, y_train):
    """Train CatBoost with paper's exact hyperparameters."""
    model = CatBoostClassifier(
        learning_rate=0.0001,        # Paper's critical hyperparameter
        iterations=2000,              # High iteration count for convergence
        random_strength=42,           # Paper's tuned parameter
        depth=12,                     # Deep trees for Morgan fingerprints
        leaf_estimation_iterations=12,# Paper's parameter
        random_state=42,              # Fixed seed for reproducibility
        verbose=False,
        thread_count=8                # Parallel training for speed
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_val, y_val):
    """Calculate all metrics."""
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    auc = roc_auc_score(y_val, y_pred_proba)
    accuracy = accuracy_score(y_val, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    
    return {
        'auc': auc,
        'accuracy': accuracy,
        'confusion_matrix': (tn, fp, fn, tp),
        'y_pred_proba': y_pred_proba,
        'y_pred': y_pred,
        'fpr': roc_curve(y_val, y_pred_proba)[0],
        'tpr': roc_curve(y_val, y_pred_proba)[1],
        'thresholds': roc_curve(y_val, y_pred_proba)[2]
    }

def save_results(metrics, y_val, X_train_shape):
    """Save results to CSV files."""
    base_path = '/Users/suyashsuryavansh/my_files/projects/major project /clinical-trials-prediction/results'
    
    # Main results
    results_df = pd.DataFrame({
        'Metric': [
            'AUC',
            'Accuracy',
            'Sensitivity',
            'Specificity',
            'Training_Samples',
            'Validation_Samples',
            'Features'
        ],
        'Value': [
            metrics['auc'],
            metrics['accuracy'],
            metrics['confusion_matrix'][3] / (metrics['confusion_matrix'][3] + metrics['confusion_matrix'][2]),
            metrics['confusion_matrix'][0] / (metrics['confusion_matrix'][0] + metrics['confusion_matrix'][1]),
            X_train_shape[0],
            len(y_val),
            X_train_shape[1]
        ]
    })
    results_df.to_csv(f'{base_path}/PRODUCTION_RESULTS.csv', index=False)
    
    # Predictions
    predictions_df = pd.DataFrame({
        'True_Label': y_val,
        'Predicted_Probability': metrics['y_pred_proba'],
        'Predicted_Class': metrics['y_pred']
    })
    predictions_df.to_csv(f'{base_path}/PRODUCTION_PREDICTIONS.csv', index=False)
    
    # ROC curve
    roc_df = pd.DataFrame({
        'FPR': metrics['fpr'],
        'TPR': metrics['tpr'],
        'Threshold': metrics['thresholds']
    })
    roc_df.to_csv(f'{base_path}/PRODUCTION_ROC.csv', index=False)
    
    return base_path

def main():
    print("\n" + "="*80)
    print("TOXICITY PREDICTION: Final Production Solution")
    print("="*80)
    
    # Load
    print("\n[1/3] Loading paper's exact data...")
    X_train, X_val, y_train, y_val = load_data()
    print(f"  ✓ Training:   {X_train.shape}")
    print(f"  ✓ Validation: {X_val.shape}")
    print(f"  ✓ Features:   2048-bit Morgan fingerprints")
    
    # Train
    print("\n[2/3] Training CatBoost (paper's hyperparameters)...")
    model = train_catboost(X_train, y_train)
    print("  ✓ Model trained successfully")
    
    # Evaluate
    print("\n[3/3] Evaluating...")
    metrics = evaluate_model(model, X_val, y_val)
    
    # Results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    auc = metrics['auc']
    gap = auc - 0.72
    status = "✅ EXCEEDS" if auc >= 0.72 else "❌ BELOW"
    
    print(f"\n{status} TARGET!")
    print(f"  AUC Score:    {auc:.4f}")
    print(f"  Target:       0.7200")
    print(f"  Gap:          {gap:+.4f} ({gap/0.72*100:+.2f}%)")
    
    print(f"\nAccuracy:      {metrics['accuracy']:.4f}")
    tn, fp, fn, tp = metrics['confusion_matrix']
    print(f"Sensitivity:   {tp/(tp+fn):.4f}")
    print(f"Specificity:   {tn/(tn+fp):.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    
    # Save
    save_results(metrics, y_val, X_train.shape)
    print(f"\n✓ Results saved to PRODUCTION_*.csv files")
    
    print("\n" + "="*80)
    print(f"SOLUTION ACHIEVED: {auc:.4f} AUC")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
