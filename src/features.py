"""
features.py - Feature engineering and preprocessing pipeline for Credit Scoring.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def engineer_financial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derives domain-specific financial ratios and risk indicators from raw credit history.
    """
    df = df.copy()

    # 1. Debt-To-Income (DTI) including proposed loan installment
    monthly_income = np.maximum(df['annual_income'] / 12.0, 1.0)
    monthly_loan_installment = df['loan_amount'] / np.maximum(df['loan_term_months'], 1)
    df['total_monthly_obligations'] = df['monthly_debt_payments'] + monthly_loan_installment
    df['debt_to_income_ratio'] = df['total_monthly_obligations'] / monthly_income

    # 2. Loan-To-Income (LTI) ratio
    df['loan_to_income_ratio'] = df['loan_amount'] / np.maximum(df['annual_income'], 1.0)

    # 3. Monthly Disposable Income after debt obligations
    df['monthly_disposable_income'] = monthly_income - df['total_monthly_obligations']

    # 4. Total Debt Burden Ratio
    df['debt_burden_ratio'] = df['total_debt'] / np.maximum(df['annual_income'], 1.0)

    # 5. Delinquency Severity Index (exponential penalty for severe late payments)
    df['delinquency_severity_index'] = (
        1.0 * df['num_late_payments_30_59_days'] +
        2.5 * df['num_late_payments_60_89_days'] +
        5.0 * df['num_late_payments_90plus_days']
    )

    # 6. Any Derogatory Event Flag
    df['has_derogatory_flag'] = (
        (df['num_late_payments_30_59_days'] > 0) |
        (df['num_late_payments_60_89_days'] > 0) |
        (df['num_late_payments_90plus_days'] > 0) |
        (df['bankruptcy_history'] > 0) |
        (df['prior_default'] > 0)
    ).astype(int)

    # 7. Payment Reliability Score: reward long credit history, penalize delinquencies
    df['payment_reliability_score'] = df['credit_history_length_months'] / (
        1.0 + df['delinquency_severity_index'] * 8.0
    )

    # 8. Credit Inquiry Density (recent inquiries relative to existing open credit lines)
    df['credit_inquiry_density'] = df['num_credit_inquiries_last_6m'] / (
        df['num_open_credit_lines'] + 1.0
    )

    # 9. Utilization Risk Flag (>75% credit card utilization)
    df['utilization_risk_flag'] = (df['credit_card_utilization'] > 0.75).astype(int)

    # 10. Credit Age Ratio (credit history length in years relative to adult age)
    adult_years = np.maximum(df['age'] - 18, 1)
    df['credit_age_ratio'] = (df['credit_history_length_months'] / 12.0) / adult_years

    # 11. Income Stability Ratio
    df['income_stability_ratio'] = df['employment_length_years'] / adult_years

    return df


def get_feature_columns() -> Tuple[List[str], List[str]]:
    """
    Returns lists of numerical and categorical feature column names.
    """
    numeric_features = [
        'age',
        'annual_income',
        'employment_length_years',
        'loan_amount',
        'loan_term_months',
        'monthly_debt_payments',
        'total_debt',
        'credit_card_utilization',
        'num_open_credit_lines',
        'num_credit_inquiries_last_6m',
        'credit_history_length_months',
        'num_late_payments_30_59_days',
        'num_late_payments_60_89_days',
        'num_late_payments_90plus_days',
        'bankruptcy_history',
        'prior_default',
        # Engineered features
        'total_monthly_obligations',
        'debt_to_income_ratio',
        'loan_to_income_ratio',
        'monthly_disposable_income',
        'debt_burden_ratio',
        'delinquency_severity_index',
        'has_derogatory_flag',
        'payment_reliability_score',
        'credit_inquiry_density',
        'utilization_risk_flag',
        'credit_age_ratio',
        'income_stability_ratio',
    ]

    categorical_features = [
        'home_ownership',
        'loan_purpose',
    ]

    return numeric_features, categorical_features


def create_preprocessor() -> ColumnTransformer:
    """
    Constructs a ColumnTransformer that scales numerical features
    and one-hot encodes categorical variables.
    """
    numeric_features, categorical_features = get_feature_columns()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
        ],
        remainder='drop'
    )
    return preprocessor
