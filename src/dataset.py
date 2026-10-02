"""
dataset.py - Dataset generation and loading utilities for Credit Scoring Model.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple


def generate_credit_data(
    n_samples: int = 10000,
    random_state: int = 42,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Generates a realistic financial dataset for creditworthiness assessment.
    
    Includes income, debts, credit card utilization, loan terms, and payment history.
    The target variable `creditworthy` (1 = Good Credit / Creditworthy, 0 = High Risk / Default)
    is probabilistically determined using genuine credit risk logic with realistic noise.
    """
    np.random.seed(random_state)

    # 1. Demographics & Stability
    age = np.random.randint(21, 72, size=n_samples)
    
    # Employment length is bounded by age
    max_emp = np.maximum(0, age - 20)
    employment_length = np.round(np.random.beta(1.5, 3.0, size=n_samples) * max_emp, 1)

    home_ownership = np.random.choice(
        ['RENT', 'MORTGAGE', 'OWN'],
        size=n_samples,
        p=[0.42, 0.46, 0.12]
    )

    # Annual income: log-normal distribution (approx $20k to $250k)
    annual_income = np.random.lognormal(mean=11.1, sigma=0.55, size=n_samples)
    annual_income = np.clip(np.round(annual_income, -2), 18000, 300000)

    # 2. Loan Details
    loan_purpose = np.random.choice(
        ['debt_consolidation', 'credit_card', 'home_improvement', 'small_business', 'major_purchase', 'medical', 'auto'],
        size=n_samples,
        p=[0.45, 0.22, 0.12, 0.07, 0.06, 0.05, 0.03]
    )
    loan_term_months = np.random.choice([36, 60], size=n_samples, p=[0.70, 0.30])
    
    # Loan amount proportional to income with variance
    loan_ratio = np.random.uniform(0.05, 0.45, size=n_samples)
    loan_amount = np.round(annual_income * loan_ratio, -2)
    loan_amount = np.clip(loan_amount, 1000, 50000)

    # 3. Debts & Credit Utilization
    # Monthly debt obligations (auto loans, student loans, mortgage/rent payments)
    base_dti = np.random.beta(2.5, 4.5, size=n_samples) * 0.65  # typically 0.05 to 0.55
    monthly_debt = np.round((annual_income / 12.0) * base_dti, 2)
    total_debt = np.round(monthly_debt * np.random.uniform(15, 60, size=n_samples), 2)

    # Revolving credit card utilization (0.0 to 1.15)
    credit_card_utilization = np.random.beta(2.0, 3.5, size=n_samples) * 1.1
    credit_card_utilization = np.clip(np.round(credit_card_utilization, 3), 0.01, 1.25)

    num_open_credit_lines = np.random.poisson(lam=8, size=n_samples)
    num_open_credit_lines = np.clip(num_open_credit_lines, 1, 30)

    num_credit_inquiries_last_6m = np.random.poisson(lam=1.2, size=n_samples)
    num_credit_inquiries_last_6m = np.clip(num_credit_inquiries_last_6m, 0, 10)

    credit_history_length_months = np.clip(
        np.round((age - 18) * 12 * np.random.uniform(0.3, 0.95, size=n_samples)),
        12, 550
    ).astype(int)

    # 4. Payment History & Delinquency
    # Delinquency rates: most applicants have 0, but some have late payments
    prob_late = 0.25
    has_lates = np.random.rand(n_samples) < prob_late
    
    num_late_30_59 = np.where(has_lates, np.random.poisson(lam=0.8, size=n_samples), 0)
    num_late_60_89 = np.where(has_lates, np.random.poisson(lam=0.3, size=n_samples), 0)
    num_late_90plus = np.where(has_lates, np.random.poisson(lam=0.15, size=n_samples), 0)

    # Severe derogatory events
    bankruptcy_prob = np.where(num_late_90plus > 0, 0.15, 0.03)
    bankruptcy_history = (np.random.rand(n_samples) < bankruptcy_prob).astype(int)

    prior_default_prob = np.where(num_late_90plus > 0, 0.20, 0.04)
    prior_default = (np.random.rand(n_samples) < prior_default_prob).astype(int)

    # 5. Determine Creditworthiness / Default Probability via Realistic Credit Scoring Function
    # Log-odds of default (higher logit -> higher risk of default)
    # Debt-to-Income (DTI)
    dti = (monthly_debt + (loan_amount / loan_term_months)) / (annual_income / 12.0)
    
    logit = (
        -2.30  # Baseline intercept (~15-20% default rate baseline)
        + 2.8 * (dti - 0.35)
        + 2.2 * (credit_card_utilization - 0.40)
        + 0.35 * num_credit_inquiries_last_6m
        + 0.50 * num_late_30_59
        + 0.90 * num_late_60_89
        + 1.60 * num_late_90plus
        + 1.50 * bankruptcy_history
        + 1.80 * prior_default
        - 0.04 * employment_length
        - 0.003 * (credit_history_length_months - 120)
        - 0.40 * (annual_income > 80000).astype(int)
        + 0.35 * (home_ownership == 'RENT').astype(int)
        - 0.25 * (home_ownership == 'OWN').astype(int)
        + np.random.normal(0, 0.45, size=n_samples)  # Unobserved idiosyncratic variance
    )

    prob_default = 1.0 / (1.0 + np.exp(-logit))
    # Default event: 1 = defaulted (bad), 0 = paid back (good)
    is_default = (np.random.rand(n_samples) < prob_default).astype(int)
    
    # In credit scoring standard conventions:
    # Target `creditworthy`: 1 = Creditworthy (non-default), 0 = Non-creditworthy (default)
    creditworthy = 1 - is_default

    df = pd.DataFrame({
        'age': age,
        'annual_income': annual_income,
        'employment_length_years': employment_length,
        'home_ownership': home_ownership,
        'loan_amount': loan_amount,
        'loan_purpose': loan_purpose,
        'loan_term_months': loan_term_months,
        'monthly_debt_payments': monthly_debt,
        'total_debt': total_debt,
        'credit_card_utilization': credit_card_utilization,
        'num_open_credit_lines': num_open_credit_lines,
        'num_credit_inquiries_last_6m': num_credit_inquiries_last_6m,
        'credit_history_length_months': credit_history_length_months,
        'num_late_payments_30_59_days': num_late_30_59,
        'num_late_payments_60_89_days': num_late_60_89,
        'num_late_payments_90plus_days': num_late_90plus,
        'bankruptcy_history': bankruptcy_history,
        'prior_default': prior_default,
        'creditworthy': creditworthy
    })

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Dataset saved to: {output_path} ({n_samples} samples)")

    return df


def load_dataset(file_path: str) -> pd.DataFrame:
    """Loads dataset from CSV file."""
    df = pd.read_csv(file_path)
    return df
