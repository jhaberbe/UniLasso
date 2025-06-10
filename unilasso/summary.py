import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def coef_evaluation_table(true_coef, est_coef, threshold=1e-8):
    # Convert to binary: 1 if non-zero, 0 if zero (or below threshold)
    true_binary = np.abs(true_coef) > threshold
    est_binary = np.abs(est_coef) > threshold

    # Confusion matrix to get TN, FP, FN, TP
    tn, fp, fn, tp = confusion_matrix(true_binary, est_binary, labels=[0, 1]).ravel()

    # Compute metrics
    ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan  # precision
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan  # sensitivity
    f1 = f1_score(true_binary, est_binary)

    # Rank correlation
    rank_corr, _ = spearmanr(true_coef, est_coef)

    # Return as a table
    df = pd.DataFrame([{
        'PPV': ppv,
        'NPV': npv,
        'Recall': recall,
        'F1': f1,
        'Spearman Rank Correlation': rank_corr
    }])

    return df

