"""
test_pipeline.py - Unit tests for Credit Scoring Model components.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd

from src.dataset import generate_credit_data
from src.features import engineer_financial_features, get_feature_columns
from src.predict import (
    predict_single_applicant,
    predict_batch,
    map_probability_to_credit_score,
    get_credit_tier
)


class TestCreditScoring(unittest.TestCase):

    def setUp(self):
        self.sample_applicant = {
            'age': 38,
            'annual_income': 85000,
            'employment_length_years': 8.5,
            'home_ownership': 'MORTGAGE',
            'loan_amount': 15000,
            'loan_purpose': 'debt_consolidation',
            'loan_term_months': 36,
            'monthly_debt_payments': 1200.0,
            'total_debt': 45000.0,
            'credit_card_utilization': 0.28,
            'num_open_credit_lines': 9,
            'num_credit_inquiries_last_6m': 1,
            'credit_history_length_months': 140,
            'num_late_payments_30_59_days': 0,
            'num_late_payments_60_89_days': 0,
            'num_late_payments_90plus_days': 0,
            'bankruptcy_history': 0,
            'prior_default': 0
        }

    def test_dataset_generation(self):
        df = generate_credit_data(n_samples=50, random_state=123)
        self.assertEqual(len(df), 50)
        self.assertIn('creditworthy', df.columns)
        self.assertTrue(set(df['creditworthy'].unique()).issubset({0, 1}))

    def test_feature_engineering(self):
        raw_df = pd.DataFrame([self.sample_applicant])
        feat_df = engineer_financial_features(raw_df)
        
        self.assertIn('debt_to_income_ratio', feat_df.columns)
        self.assertIn('loan_to_income_ratio', feat_df.columns)
        self.assertIn('delinquency_severity_index', feat_df.columns)
        self.assertIn('payment_reliability_score', feat_df.columns)
        self.assertGreater(feat_df['payment_reliability_score'].iloc[0], 0)

    def test_credit_score_mapping(self):
        score_high = map_probability_to_credit_score(0.95)
        score_low = map_probability_to_credit_score(0.05)
        
        self.assertTrue(300 <= score_high <= 850)
        self.assertTrue(300 <= score_low <= 850)
        self.assertGreater(score_high, score_low)

        tier_high = get_credit_tier(score_high)
        self.assertIn('tier', tier_high)
        self.assertIn(tier_high['tier'], ['Excellent', 'Good'])

    def test_single_prediction(self):
        res = predict_single_applicant(
            self.sample_applicant,
            pipeline_path="models/Random_Forest_pipeline.joblib"
        )
        self.assertIn('creditworthy', res)
        self.assertIn('decision', res)
        self.assertIn('credit_score', res)
        self.assertIn('risk_tier', res)
        self.assertTrue(300 <= res['credit_score'] <= 850)
        self.assertIsInstance(res['positive_factors'], list)

    def test_batch_prediction(self):
        df = generate_credit_data(n_samples=10, random_state=42)
        scored_df = predict_batch(df, pipeline_path="models/Random_Forest_pipeline.joblib")
        
        self.assertIn('credit_score', scored_df.columns)
        self.assertIn('decision', scored_df.columns)
        self.assertEqual(len(scored_df), 10)


if __name__ == '__main__':
    unittest.main()
