"""
evaluate.py - Model accuracy assessment and performance visualization.
Computes Precision, Recall, F1-Score, ROC-AUC, PR-AUC, Confusion Matrix, etc.
"""

import os
import json
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    brier_score_loss
)


def evaluate_model_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray
) -> Dict[str, Any]:
    """
    Computes a comprehensive suite of credit scoring classification metrics.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Specificity = True Negative Rate = TN / (TN + FP)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'specificity': float(specificity),
        'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'roc_auc': float(roc_auc_score(y_true, y_prob)),
        'pr_auc': float(average_precision_score(y_true, y_prob)),
        'brier_score': float(brier_score_loss(y_true, y_prob)),
        'confusion_matrix': {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        },
        'classification_report': classification_report(y_true, y_pred, output_dict=True)
    }
    return metrics


def plot_combined_roc_curves(
    model_predictions: Dict[str, np.ndarray],
    y_true: np.ndarray,
    output_path: str
):
    """
    Plots ROC curves for all models side-by-side on a single graph.
    """
    plt.figure(figsize=(9, 7))
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    for model_name, y_prob in model_predictions.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        plt.plot(fpr, tpr, lw=2.2, label=f"{model_name} (AUC = {auc:.3f})")

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.5, label='Random Chance (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate (Recall / Sensitivity)', fontsize=12, fontweight='bold')
    plt.title('Receiver Operating Characteristic (ROC) Comparison', fontsize=14, fontweight='bold', pad=15)
    plt.legend(loc="lower right", fontsize=11, frameon=True)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"ROC Curve plot saved to: {output_path}")


def plot_combined_pr_curves(
    model_predictions: Dict[str, np.ndarray],
    y_true: np.ndarray,
    output_path: str
):
    """
    Plots Precision-Recall curves for all models.
    """
    plt.figure(figsize=(9, 7))
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    for model_name, y_prob in model_predictions.items():
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        plt.plot(recall, precision, lw=2.2, label=f"{model_name} (PR-AUC = {ap:.3f})")

    baseline = np.mean(y_true)
    plt.axhline(y=baseline, color='gray', linestyle='--', lw=1.5, label=f'Baseline ({baseline:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12, fontweight='bold')
    plt.ylabel('Precision', fontsize=12, fontweight='bold')
    plt.title('Precision-Recall Curve Comparison', fontsize=14, fontweight='bold', pad=15)
    plt.legend(loc="lower left", fontsize=11, frameon=True)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"PR Curve plot saved to: {output_path}")


def plot_confusion_matrices(
    cms: Dict[str, np.ndarray],
    output_path: str
):
    """
    Plots a multi-panel grid of confusion matrices.
    """
    n_models = len(cms)
    cols = 2
    rows = (n_models + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
    axes = np.array(axes).flatten()

    for idx, (model_name, cm) in enumerate(cms.items()):
        ax = axes[idx]
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            cbar=False,
            ax=ax,
            annot_kws={'size': 14, 'weight': 'bold'},
            xticklabels=['Default (0)', 'Creditworthy (1)'],
            yticklabels=['Default (0)', 'Creditworthy (1)']
        )
        ax.set_title(f"{model_name} Confusion Matrix", fontsize=13, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=11)
        ax.set_ylabel('True Label', fontsize=11)

    # Hide extra subplots if any
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Confusion Matrix grid saved to: {output_path}")


def plot_metrics_comparison(
    metrics_summary: pd.DataFrame,
    output_path: str
):
    """
    Plots a grouped bar chart comparing Accuracy, Precision, Recall, F1, and ROC-AUC.
    """
    plt.figure(figsize=(11, 6))
    
    # Reshape for seaborn
    plot_df = metrics_summary.melt(
        id_vars=['Model'],
        value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
        var_name='Metric',
        value_name='Score'
    )

    palette = ['#2563eb', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899']
    ax = sns.barplot(x='Model', y='Score', hue='Metric', data=plot_df, palette=palette)
    plt.ylim([0.5, 1.02])
    plt.title('Model Accuracy Assessment & Metrics Comparison', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Classification Algorithm', fontsize=12, fontweight='bold')
    plt.ylabel('Metric Score', fontsize=12, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, fontsize=10)

    # Annotate bar values
    for p in ax.patches:
        height = p.get_height()
        if height > 0.1:
            ax.annotate(f'{height:.2f}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom',
                        fontsize=8, color='black',
                        xytext=(0, 2), textcoords='offset points')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Metrics Comparison bar chart saved to: {output_path}")


def plot_feature_importance(
    fi_df: pd.DataFrame,
    model_name: str,
    output_path: str
):
    """
    Plots horizontal bar chart of top features.
    """
    metric_col = [c for c in fi_df.columns if c != 'Feature'][0]
    plt.figure(figsize=(9, 6))
    
    sns.barplot(
        x=metric_col,
        y='Feature',
        data=fi_df.sort_values(by=metric_col, ascending=True),
        palette='viridis'
    )
    plt.title(f'{model_name} - Top Feature Importance', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel(metric_col, fontsize=11, fontweight='bold')
    plt.ylabel('Financial Feature', fontsize=11, fontweight='bold')
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
