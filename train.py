"""
train.py - Orchestrates data preparation, feature engineering, model training,
accuracy assessment, and model persistence.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.dataset import generate_credit_data
from src.features import engineer_financial_features
from src.models import (
    get_model_definitions,
    train_model_pipeline,
    extract_feature_importance,
    save_pipeline
)
from src.evaluate import (
    evaluate_model_performance,
    plot_combined_roc_curves,
    plot_combined_pr_curves,
    plot_confusion_matrices,
    plot_metrics_comparison,
    plot_feature_importance
)


def run_training_pipeline(
    data_path: str = "data/credit_data.csv",
    models_dir: str = "models",
    reports_dir: str = "reports",
    n_samples: int = 10000,
    random_state: int = 42
):
    print("=" * 65)
    print("           CREDIT SCORING MODEL TRAINING PIPELINE")
    print("=" * 65)

    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # 1. Dataset Generation or Loading
    if os.path.exists(data_path):
        print(f"\n[1/5] Loading existing dataset from: {data_path}")
        df = pd.read_csv(data_path)
    else:
        print(f"\n[1/5] Generating {n_samples:,} synthetic financial records...")
        df = generate_credit_data(n_samples=n_samples, random_state=random_state, output_path=data_path)

    print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    target_dist = df['creditworthy'].value_counts(normalize=True)
    print(f"Target distribution: Creditworthy (1): {target_dist.get(1, 0):.1%}, High Risk Default (0): {target_dist.get(0, 0):.1%}")

    # 2. Feature Engineering
    print("\n[2/5] Engineering financial ratios and risk metrics...")
    df_engineered = engineer_financial_features(df)
    print(f"Post-engineering features: {df_engineered.shape[1]} total columns")

    # 3. Train-Test Split (Stratified)
    X = df_engineered.drop(columns=['creditworthy'])
    y = df_engineered['creditworthy']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=random_state,
        stratify=y
    )
    print(f"Training samples: {len(X_train):,}, Testing samples: {len(X_test):,}")

    # 4. Model Training & Evaluation
    print("\n[3/5] Training classification algorithms...")
    model_defs = get_model_definitions(random_state=random_state)
    
    evaluation_records = []
    test_prob_dict = {}
    test_cm_dict = {}
    trained_pipelines = {}

    for name, estimator in model_defs.items():
        print(f"\n--- Training: {name} ---")
        pipeline = train_model_pipeline(name, estimator, X_train, y_train)
        trained_pipelines[name] = pipeline

        # Predictions
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        test_prob_dict[name] = y_prob

        # Evaluation metrics
        metrics = evaluate_model_performance(y_test.values, y_pred, y_prob)
        test_cm_dict[name] = np.array([
            [metrics['confusion_matrix']['true_negatives'], metrics['confusion_matrix']['false_positives']],
            [metrics['confusion_matrix']['false_negatives'], metrics['confusion_matrix']['true_positives']]
        ])

        # Record summary
        evaluation_records.append({
            'Model': name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'Specificity': metrics['specificity'],
            'F1-Score': metrics['f1_score'],
            'ROC-AUC': metrics['roc_auc'],
            'PR-AUC': metrics['pr_auc'],
            'Brier Score': metrics['brier_score']
        })

        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

        # Save individual model
        clean_name = name.replace(" ", "_")
        model_save_path = os.path.join(models_dir, f"{clean_name}_pipeline.joblib")
        save_pipeline(pipeline, model_save_path)

        # Feature Importance
        fi_df = extract_feature_importance(pipeline, top_n=12)
        if not fi_df.empty:
            fi_plot_path = os.path.join(reports_dir, f"feature_importance_{clean_name}.png")
            plot_feature_importance(fi_df, name, fi_plot_path)

    # 5. Generate Visualizations & Reports
    print("\n[4/5] Generating evaluation plots & performance benchmarks...")
    metrics_df = pd.DataFrame(evaluation_records)
    metrics_csv_path = os.path.join(reports_dir, "metrics_summary.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"Saved metrics CSV to: {metrics_csv_path}")

    # Metrics JSON
    with open(os.path.join(reports_dir, "metrics_summary.json"), 'w') as f:
        json.dump(evaluation_records, f, indent=4)

    # Plots
    plot_combined_roc_curves(
        test_prob_dict,
        y_test.values,
        os.path.join(reports_dir, "roc_curves.png")
    )

    plot_combined_pr_curves(
        test_prob_dict,
        y_test.values,
        os.path.join(reports_dir, "precision_recall_curves.png")
    )

    plot_confusion_matrices(
        test_cm_dict,
        os.path.join(reports_dir, "confusion_matrices.png")
    )

    plot_metrics_comparison(
        metrics_df,
        os.path.join(reports_dir, "metrics_comparison.png")
    )

    print("\n[5/5] Pipeline execution complete!")
    print("\n" + "=" * 65)
    print("                    BENCHMARK RESULTS")
    print("=" * 65)
    print(metrics_df.to_string(index=False))
    print("=" * 65)


if __name__ == '__main__':
    run_training_pipeline()
