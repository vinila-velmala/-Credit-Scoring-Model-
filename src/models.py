"""
models.py - Model definitions, training, and persistence for Credit Scoring.
"""

import os
import joblib
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline

from src.features import create_preprocessor, engineer_financial_features


def get_model_definitions(random_state: int = 42) -> Dict[str, Any]:
    """
    Returns a dictionary of classification algorithms suited for credit scoring.
    """
    models = {
        'Logistic Regression': LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight='balanced',
            random_state=random_state,
            solver='lbfgs'
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=20,
            class_weight='balanced',
            random_state=random_state
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=120,
            learning_rate=0.08,
            max_depth=4,
            subsample=0.85,
            random_state=random_state
        )
    }
    return models


def train_model_pipeline(
    model_name: str,
    estimator: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> Pipeline:
    """
    Wraps the preprocessor and estimator into a Scikit-Learn Pipeline and fits it on X_train.
    """
    preprocessor = create_preprocessor()
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', estimator)
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def extract_feature_importance(pipeline: Pipeline, top_n: int = 15) -> pd.DataFrame:
    """
    Extracts feature importances (or absolute coefficients) from a trained pipeline.
    """
    preprocessor = pipeline.named_steps['preprocessor']
    classifier = pipeline.named_steps['classifier']

    # Get transformed feature names
    cat_encoder = preprocessor.named_transformers_['cat']
    cat_cols = preprocessor.transformers[1][2]
    cat_feature_names = cat_encoder.get_feature_names_out(cat_cols)
    num_feature_names = preprocessor.transformers[0][2]
    all_feature_names = list(num_feature_names) + list(cat_feature_names)

    if hasattr(classifier, 'feature_importances_'):
        importances = classifier.feature_importances_
        metric_name = 'Importance'
    elif hasattr(classifier, 'coef_'):
        importances = np.abs(classifier.coef_[0])
        metric_name = 'Abs_Coefficient'
    else:
        return pd.DataFrame()

    fi_df = pd.DataFrame({
        'Feature': all_feature_names,
        metric_name: importances
    }).sort_values(by=metric_name, ascending=False).head(top_n)

    return fi_df


def save_pipeline(pipeline: Pipeline, model_path: str):
    """Serializes pipeline to disk."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"Saved pipeline to: {model_path}")


def load_pipeline(model_path: str) -> Pipeline:
    """Loads serialized pipeline from disk."""
    return joblib.load(model_path)
