"""
predict.py - Inference pipeline for predicting individual creditworthiness,
calculating estimated credit score (300-850), and diagnosing risk factors.
"""

import os
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import joblib

from src.features import engineer_financial_features


def map_probability_to_credit_score(prob_creditworthy: float) -> int:
    """
    Maps creditworthy probability (0.0 to 1.0) into standard credit score range (300 - 850).
    """
    # Using non-linear mapping mimicking FICO distribution
    # 0.50 probability maps to ~630 (Fair)
    score = 300 + (prob_creditworthy ** 0.85) * 550
    return int(np.clip(round(score), 300, 850))


def get_credit_tier(score: int) -> Dict[str, str]:
    """
    Classifies credit score into standard financial risk tiers.
    """
    if score >= 750:
        return {'tier': 'Excellent', 'color': '#10b981', 'badge': 'LOW RISK'}
    elif score >= 670:
        return {'tier': 'Good', 'color': '#3b82f6', 'badge': 'MODERATE-LOW RISK'}
    elif score >= 580:
        return {'tier': 'Fair', 'color': '#f59e0b', 'badge': 'MODERATE RISK'}
    else:
        return {'tier': 'Poor', 'color': '#ef4444', 'badge': 'HIGH RISK / SUBPRIME'}


def diagnose_applicant(record: pd.Series) -> Tuple[List[str], List[str]]:
    """
    Identifies key strengths and risk flags for explainable credit decisions.
    """
    positive_factors = []
    risk_factors = []

    # DTI check
    dti = record.get('debt_to_income_ratio', 0)
    if dti <= 0.25:
        positive_factors.append(f"Low Debt-to-Income ratio ({dti:.1%}) indicates strong debt servicing capacity.")
    elif dti > 0.45:
        risk_factors.append(f"High Debt-to-Income ratio ({dti:.1%}) exceeds recommended 43% lending ceiling.")

    # Utilization
    util = record.get('credit_card_utilization', 0)
    if util < 0.25:
        positive_factors.append(f"Healthy revolving credit utilization ({util:.1%}) well below standard thresholds.")
    elif util > 0.65:
        risk_factors.append(f"Elevated credit card utilization ({util:.1%}) signals potential liquidity strain.")

    # Delinquency
    late_30 = record.get('num_late_payments_30_59_days', 0)
    late_60 = record.get('num_late_payments_60_89_days', 0)
    late_90 = record.get('num_late_payments_90plus_days', 0)
    tot_late = late_30 + late_60 + late_90

    if tot_late == 0:
        positive_factors.append("Clean payment record with zero past-due delinquencies.")
    else:
        risk_factors.append(f"Payment history includes {tot_late} late payment incident(s) (90+ day lates: {late_90}).")

    if record.get('bankruptcy_history', 0) == 1:
        risk_factors.append("Public record derogatory event: Prior bankruptcy recorded.")
    if record.get('prior_default', 0) == 1:
        risk_factors.append("Derogatory event: Historical loan default on record.")

    # Inquiries
    inquiries = record.get('num_credit_inquiries_last_6m', 0)
    if inquiries >= 4:
        risk_factors.append(f"High number of recent credit inquiries ({inquiries} in last 6 months) indicates credit shopping.")
    elif inquiries == 0:
        positive_factors.append("No recent hard credit inquiries in the past 6 months.")

    # Credit History Length
    hist_months = record.get('credit_history_length_months', 0)
    if hist_months >= 72:
        positive_factors.append(f"Established credit history length of {hist_months // 12} years.")
    elif hist_months < 24:
        risk_factors.append(f"Limited credit track record ({hist_months} months).")

    # Income
    income = record.get('annual_income', 0)
    if income >= 75000:
        positive_factors.append(f"Robust annual earned income (${income:,.0f}).")
    elif income < 30000:
        risk_factors.append(f"Lower annual income bracket (${income:,.0f}).")

    return positive_factors, risk_factors


def predict_single_applicant(
    applicant_data: Dict[str, Any],
    pipeline_path: str = "models/Random_Forest_pipeline.joblib"
) -> Dict[str, Any]:
    """
    Scores an individual loan/credit applicant.
    """
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(f"Model pipeline not found at {pipeline_path}. Please train models first.")

    pipeline = joblib.load(pipeline_path)

    # Convert to DataFrame and apply feature engineering
    raw_df = pd.DataFrame([applicant_data])
    processed_df = engineer_financial_features(raw_df)

    prob_creditworthy = float(pipeline.predict_proba(processed_df)[0][1])
    prob_default = float(1.0 - prob_creditworthy)
    is_creditworthy = bool(prob_creditworthy >= 0.50)

    score = map_probability_to_credit_score(prob_creditworthy)
    tier_info = get_credit_tier(score)

    if prob_creditworthy >= 0.65:
        decision = "APPROVED"
        decision_notes = "Applicant meets prime credit underwriting criteria."
    elif prob_creditworthy >= 0.45:
        decision = "MANUAL REVIEW"
        decision_notes = "Borderline credit profile. Underwriter manual review recommended."
    else:
        decision = "DECLINED"
        decision_notes = "High predicted probability of default. Fails automated underwriting standard."

    row = processed_df.iloc[0]
    strengths, risks = diagnose_applicant(row)

    return {
        'creditworthy': is_creditworthy,
        'decision': decision,
        'decision_notes': decision_notes,
        'creditworthiness_probability': round(prob_creditworthy, 4),
        'default_probability': round(prob_default, 4),
        'credit_score': score,
        'risk_tier': tier_info['tier'],
        'risk_badge': tier_info['badge'],
        'risk_color': tier_info['color'],
        'positive_factors': strengths,
        'risk_factors': risks,
        'engineered_metrics': {
            'debt_to_income_ratio': round(float(row['debt_to_income_ratio']), 3),
            'loan_to_income_ratio': round(float(row['loan_to_income_ratio']), 3),
            'monthly_disposable_income': round(float(row['monthly_disposable_income']), 2),
            'delinquency_severity_index': round(float(row['delinquency_severity_index']), 2),
            'payment_reliability_score': round(float(row['payment_reliability_score']), 2)
        }
    }


def predict_batch(
    df: pd.DataFrame,
    pipeline_path: str = "models/Random_Forest_pipeline.joblib"
) -> pd.DataFrame:
    """
    Evaluates a batch of applicants and appends scoring columns.
    """
    pipeline = joblib.load(pipeline_path)
    processed_df = engineer_financial_features(df)
    
    probs = pipeline.predict_proba(processed_df)[:, 1]
    
    output_df = df.copy()
    output_df['prob_creditworthy'] = np.round(probs, 4)
    output_df['prob_default'] = np.round(1.0 - probs, 4)
    output_df['credit_score'] = [map_probability_to_credit_score(p) for p in probs]
    output_df['risk_tier'] = [get_credit_tier(s)['tier'] for s in output_df['credit_score']]
    output_df['decision'] = [
        "APPROVED" if p >= 0.65 else ("MANUAL REVIEW" if p >= 0.45 else "DECLINED")
        for p in probs
    ]
    return output_df
