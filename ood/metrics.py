import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
import logging

def compute_fpr95(id_scores, ood_scores):
    """Compute FPR at 95% TPR"""
    try:
        # Create labels: 1 for ID, 0 for OOD
        y_true = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
        y_scores = np.concatenate([id_scores, ood_scores])
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        fpr95_idx = np.argmax(tpr >= 0.95)
        fpr95 = fpr[fpr95_idx] * 100 if fpr95_idx < len(fpr) else 100.0
        return fpr95
    except:
        return 100.0

def compute_auroc(id_scores, ood_scores):
    """Compute AUROC"""
    try:
        y_true = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
        y_scores = np.concatenate([id_scores, ood_scores])
        return roc_auc_score(y_true, y_scores) * 100
    except:
        return 50.0

def compute_ood_metrics(id_scores, ood_scores, method_name="OOD"):
    """Compute comprehensive OOD metrics"""
    try:
        auroc = compute_auroc(id_scores, ood_scores)
        fpr95 = compute_fpr95(id_scores, ood_scores)
        
        return {
            'method': method_name,
            'auroc': auroc,
            'fpr95': fpr95,
            'id_samples': len(id_scores),
            'ood_samples': len(ood_scores)
        }
    except Exception as e:
        return {
            'method': method_name,
            'error': str(e)
        }

def compute_threshold_accuracy(id_scores, ood_scores, threshold):
    """Compute accuracy with given threshold"""
    try:
        # ID predictions: score >= threshold → 1 (ID), score < threshold → 0 (OOD)
        id_predictions = (np.array(id_scores) >= threshold).astype(int)
        ood_predictions = (np.array(ood_scores) >= threshold).astype(int)
        
        # True labels
        id_labels = np.ones(len(id_scores))  # All ID samples should be 1
        ood_labels = np.zeros(len(ood_scores))  # All OOD samples should be 0
        
        # Compute accuracy
        all_predictions = np.concatenate([id_predictions, ood_predictions])
        all_labels = np.concatenate([id_labels, ood_labels])
        
        accuracy = accuracy_score(all_labels, all_predictions)
        return accuracy
    except:
        return 0.0