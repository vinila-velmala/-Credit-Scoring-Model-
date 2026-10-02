"""
app.py - Interactive Credit Scoring & Risk Assessment Dashboard.
Built with Streamlit for real-time inference, model evaluation, and batch underwriting.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.features import engineer_financial_features
from src.predict import (
    predict_single_applicant,
    predict_batch,
    map_probability_to_credit_score,
    get_credit_tier
)
from src.dataset import generate_credit_data

# Page setup
st.set_page_config(
    page_title="CreditIQ - Intelligent Credit Scoring System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1e3a8a, #3b82f6, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .score-banner {
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 1.2rem;
    }
    .factor-pill-pos {
        background-color: #ecfdf5;
        border-left: 4px solid #10b981;
        padding: 0.75rem 1rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
        font-size: 0.95rem;
        color: #065f46;
    }
    .factor-pill-risk {
        background-color: #fef2f2;
        border-left: 4px solid #ef4444;
        padding: 0.75rem 1rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
        font-size: 0.95rem;
        color: #991b1b;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-title">💳 CreditIQ: Automated Credit Scoring Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Predicting creditworthiness and default probabilities via machine learning classification and financial feature engineering.</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Single Applicant Underwriting",
    "📊 Model Performance & Accuracy",
    "⚙️ Feature Engineering & Dataset",
    "📁 Batch Applicant Scoring"
])

# -------------------------------------------------------------
# TAB 1: Real-time Single Applicant Prediction
# -------------------------------------------------------------
with tab1:
    st.subheader("Interactive Credit Underwriting Simulator")
    st.markdown("Adjust the financial and credit history variables below to calculate the real-time credit score, default probability, and underwriting decision.")

    # Preset applicant profiles
    col_presets, col_model_sel = st.columns([2, 1])
    with col_presets:
        preset = st.selectbox(
            "Quick Load Profile Preset:",
            ["Custom Input", "Prime Borrower (Low Risk)", "Near-Prime Borrower (Moderate Risk)", "Subprime Borrower (High Risk)"]
        )
    with col_model_sel:
        chosen_model = st.selectbox(
            "Underwriting Model:",
            ["Random Forest", "Gradient Boosting", "Logistic Regression", "Decision Tree"]
        )

    # Preset values
    if preset == "Prime Borrower (Low Risk)":
        d_age, d_inc, d_emp, d_home = 42, 95000, 10.0, "MORTGAGE"
        d_loan, d_purp, d_term = 18000, "home_improvement", 36
        d_mdebt, d_totdebt, d_util = 1100, 32000, 0.18
        d_lines, d_inq, d_hist = 12, 0, 180
        d_l30, d_l60, d_l90 = 0, 0, 0
        d_bank, d_def = 0, 0
    elif preset == "Near-Prime Borrower (Moderate Risk)":
        d_age, d_inc, d_emp, d_home = 31, 52000, 3.5, "RENT"
        d_loan, d_purp, d_term = 14000, "debt_consolidation", 60
        d_mdebt, d_totdebt, d_util = 1450, 24000, 0.58
        d_lines, d_inq, d_hist = 7, 2, 84
        d_l30, d_l60, d_l90 = 1, 0, 0
        d_bank, d_def = 0, 0
    elif preset == "Subprime Borrower (High Risk)":
        d_age, d_inc, d_emp, d_home = 26, 28000, 1.0, "RENT"
        d_loan, d_purp, d_term = 12000, "credit_card", 36
        d_mdebt, d_totdebt, d_util = 1200, 29000, 0.89
        d_lines, d_inq, d_hist = 5, 4, 36
        d_l30, d_l60, d_l90 = 2, 1, 1
        d_bank, d_def = 1, 0
    else:
        d_age, d_inc, d_emp, d_home = 35, 65000, 5.0, "RENT"
        d_loan, d_purp, d_term = 15000, "debt_consolidation", 36
        d_mdebt, d_totdebt, d_util = 1300, 30000, 0.42
        d_lines, d_inq, d_hist = 8, 1, 120
        d_l30, d_l60, d_l90 = 0, 0, 0
        d_bank, d_def = 0, 0

    with st.form("applicant_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 👤 Applicant & Employment")
            age = st.slider("Age (years)", 18, 75, d_age)
            annual_income = st.number_input("Annual Income ($)", min_value=12000, max_value=500000, value=d_inc, step=5000)
            employment_length_years = st.slider("Employment Length (years)", 0.0, 40.0, float(d_emp), 0.5)
            home_ownership = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN"], index=["RENT", "MORTGAGE", "OWN"].index(d_home))

        with col2:
            st.markdown("##### 💰 Loan Application Details")
            loan_amount = st.number_input("Requested Loan Amount ($)", min_value=1000, max_value=60000, value=d_loan, step=1000)
            loan_term_months = st.selectbox("Loan Term (Months)", [36, 60], index=[36, 60].index(d_term))
            loan_purpose = st.selectbox(
                "Loan Purpose",
                ['debt_consolidation', 'credit_card', 'home_improvement', 'small_business', 'major_purchase', 'medical', 'auto'],
                index=['debt_consolidation', 'credit_card', 'home_improvement', 'small_business', 'major_purchase', 'medical', 'auto'].index(d_purp)
            )
            monthly_debt_payments = st.number_input("Existing Monthly Debt Service ($)", min_value=0, max_value=15000, value=int(d_mdebt), step=100)
            total_debt = st.number_input("Total Outstanding Debt ($)", min_value=0, max_value=300000, value=int(d_totdebt), step=2500)

        with col3:
            st.markdown("##### 📜 Credit & Delinquency History")
            credit_card_utilization = st.slider("Credit Card Utilization Ratio", 0.0, 1.25, float(d_util), 0.02)
            num_open_credit_lines = st.slider("Open Credit Lines", 1, 30, d_lines)
            num_credit_inquiries_last_6m = st.slider("Credit Inquiries (Last 6 Months)", 0, 10, d_inq)
            credit_history_length_months = st.slider("Credit History Length (Months)", 12, 480, d_hist)
            num_late_payments_30_59_days = st.number_input("Late Payments (30-59 Days)", 0, 10, d_l30)
            num_late_payments_60_89_days = st.number_input("Late Payments (60-89 Days)", 0, 10, d_l60)
            num_late_payments_90plus_days = st.number_input("Late Payments (90+ Days)", 0, 10, d_l90)
            
            c_sub1, c_sub2 = st.columns(2)
            with c_sub1:
                bankruptcy_history = st.checkbox("Prior Bankruptcy", value=bool(d_bank))
            with c_sub2:
                prior_default = st.checkbox("Historical Loan Default", value=bool(d_def))

        submit_btn = st.form_submit_button("⚡ Evaluate Creditworthiness", use_container_width=True)

    if submit_btn or preset:
        applicant_payload = {
            'age': age,
            'annual_income': annual_income,
            'employment_length_years': employment_length_years,
            'home_ownership': home_ownership,
            'loan_amount': loan_amount,
            'loan_purpose': loan_purpose,
            'loan_term_months': loan_term_months,
            'monthly_debt_payments': float(monthly_debt_payments),
            'total_debt': float(total_debt),
            'credit_card_utilization': credit_card_utilization,
            'num_open_credit_lines': num_open_credit_lines,
            'num_credit_inquiries_last_6m': num_credit_inquiries_last_6m,
            'credit_history_length_months': credit_history_length_months,
            'num_late_payments_30_59_days': num_late_payments_30_59_days,
            'num_late_payments_60_89_days': num_late_payments_60_89_days,
            'num_late_payments_90plus_days': num_late_payments_90plus_days,
            'bankruptcy_history': int(bankruptcy_history),
            'prior_default': int(prior_default)
        }

        pipeline_file = f"models/{chosen_model.replace(' ', '_')}_pipeline.joblib"
        if not os.path.exists(pipeline_file):
            st.error(f"Model file `{pipeline_file}` not found. Please run `python train.py` first.")
        else:
            result = predict_single_applicant(applicant_payload, pipeline_path=pipeline_file)

            st.markdown("---")
            st.markdown("### 🏆 Underwriting Decision & Risk Profile")

            # Decision Banner
            bg_color = result['risk_color']
            st.markdown(f"""
            <div style="background-color: {bg_color}; padding: 1.5rem; border-radius: 12px; color: white; text-align: center; margin-bottom: 1.5rem;">
                <h1 style="margin: 0; color: white; font-size: 2.5rem;">{result['decision']}</h1>
                <p style="margin: 0.3rem 0 0 0; font-size: 1.25rem; font-weight: 500;">
                    Estimated FICO Score: <strong>{result['credit_score']}</strong> &nbsp;|&nbsp; Tier: <strong>{result['risk_tier']}</strong> ({result['risk_badge']})
                </p>
                <p style="margin: 0.4rem 0 0 0; font-size: 0.95rem; opacity: 0.9;">{result['decision_notes']}</p>
            </div>
            """, unsafe_allow_html=True)

            # Key metric tiles
            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                st.metric("Credit Score", f"{result['credit_score']} / 850")
            with m2:
                st.metric("Creditworthy Prob.", f"{result['creditworthiness_probability']:.1%}")
            with m3:
                st.metric("Default Risk Prob.", f"{result['default_probability']:.1%}")
            with m4:
                st.metric("Debt-to-Income (DTI)", f"{result['engineered_metrics']['debt_to_income_ratio']:.1%}")
            with m5:
                st.metric("Disposable Income", f"${result['engineered_metrics']['monthly_disposable_income']:,.0f}/mo")

            # Explainability / Diagnostic Breakdown
            st.markdown("#### 🔍 Explainable Credit Diagnostics")
            c_pos, c_risk = st.columns(2)

            with c_pos:
                st.markdown("##### 🟢 Favorable Credit Factors")
                if result['positive_factors']:
                    for pos in result['positive_factors']:
                        st.markdown(f'<div class="factor-pill-pos">✓ {pos}</div>', unsafe_allow_html=True)
                else:
                    st.info("No standout positive credit factors identified.")

            with c_risk:
                st.markdown("##### 🔴 Identified Risk Flags")
                if result['risk_factors']:
                    for risk in result['risk_factors']:
                        st.markdown(f'<div class="factor-pill-risk">⚠ {risk}</div>', unsafe_allow_html=True)
                else:
                    st.success("No critical risk flags detected.")

            # Engineered Ratios Table
            with st.expander("📊 View Detailed Engineered Financial Ratios", expanded=False):
                eng_df = pd.DataFrame([result['engineered_metrics']]).T
                eng_df.columns = ["Calculated Value"]
                st.table(eng_df)

# -------------------------------------------------------------
# TAB 2: Model Performance & Accuracy
# -------------------------------------------------------------
with tab2:
    st.subheader("Model Accuracy Assessment & Multi-Algorithm Benchmark")
    st.markdown("Comparing **Logistic Regression**, **Decision Trees**, **Random Forest**, and **Gradient Boosting** on the test partition.")

    metrics_csv_path = "reports/metrics_summary.csv"
    if os.path.exists(metrics_csv_path):
        m_df = pd.read_csv(metrics_csv_path)
        
        # Display styled metrics table
        st.dataframe(
            m_df.style.format({
                'Accuracy': '{:.2%}',
                'Precision': '{:.2%}',
                'Recall': '{:.2%}',
                'Specificity': '{:.2%}',
                'F1-Score': '{:.2%}',
                'ROC-AUC': '{:.4f}',
                'PR-AUC': '{:.4f}',
                'Brier Score': '{:.4f}'
            }).highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC'], color='#dcfce7'),
            use_container_width=True
        )

        st.markdown("---")
        st.subheader("Visual Performance Analytics")

        c_roc, c_pr = st.columns(2)
        with c_roc:
            if os.path.exists("reports/roc_curves.png"):
                st.image("reports/roc_curves.png", caption="Receiver Operating Characteristic (ROC) Comparison", use_container_width=True)
        with c_pr:
            if os.path.exists("reports/precision_recall_curves.png"):
                st.image("reports/precision_recall_curves.png", caption="Precision-Recall Curves Comparison", use_container_width=True)

        c_cm, c_comp = st.columns(2)
        with c_cm:
            if os.path.exists("reports/confusion_matrices.png"):
                st.image("reports/confusion_matrices.png", caption="Confusion Matrix Grid across Models", use_container_width=True)
        with c_comp:
            if os.path.exists("reports/metrics_comparison.png"):
                st.image("reports/metrics_comparison.png", caption="Grouped Performance Metric Comparison", use_container_width=True)

        st.markdown("---")
        st.subheader("Global Feature Importance Rankings")
        fi_cols = st.columns(3)
        with fi_cols[0]:
            if os.path.exists("reports/feature_importance_Random_Forest.png"):
                st.image("reports/feature_importance_Random_Forest.png", caption="Random Forest Feature Importance", use_container_width=True)
        with fi_cols[1]:
            if os.path.exists("reports/feature_importance_Decision_Tree.png"):
                st.image("reports/feature_importance_Decision_Tree.png", caption="Decision Tree Feature Importance", use_container_width=True)
        with fi_cols[2]:
            if os.path.exists("reports/feature_importance_Logistic_Regression.png"):
                st.image("reports/feature_importance_Logistic_Regression.png", caption="Logistic Regression Feature Weights", use_container_width=True)

    else:
        st.warning("Benchmark reports not yet generated. Please execute `python train.py` first.")

# -------------------------------------------------------------
# TAB 3: Feature Engineering & Dataset
# -------------------------------------------------------------
with tab3:
    st.subheader("Feature Engineering from Financial History")
    st.markdown("""
    In financial credit scoring, raw variables alone (like nominal income or nominal debt) are insufficient. 
    Lenders rely on **domain-specific financial ratios** that measure debt affordability, liquidity strain, and delinquency recency.
    """)

    col_fe1, col_fe2 = st.columns(2)
    with col_fe1:
        st.markdown("""
        #### 📐 Engineered Financial Indicators:
        - **Debt-To-Income (DTI)**:
          $$\\text{DTI} = \\frac{\\text{Monthly Debt Payments} + \\frac{\\text{Loan Amount}}{\\text{Loan Term}}}{\\frac{\\text{Annual Income}}{12}}$$
          Measures what fraction of monthly gross income is committed to existing and new debt service. Standard lending ceiling is ~43%.
        
        - **Loan-To-Income (LTI)**:
          $$\\text{LTI} = \\frac{\\text{Loan Amount}}{\\text{Annual Income}}$$
          Measures principal exposure relative to earning power.
        
        - **Monthly Disposable Income**:
          $$\\text{Disposable} = \\frac{\\text{Annual Income}}{12} - \\text{Total Monthly Obligations}$$
          Buffer available for unexpected financial shocks.
        """)

    with col_fe2:
        st.markdown("""
        #### 🛡️ Delinquency & Credit Strain Indices:
        - **Delinquency Severity Index**:
          $$\\text{Index} = 1.0 \\times \\text{Late}_{30-59} + 2.5 \\times \\text{Late}_{60-89} + 5.0 \\times \\text{Late}_{90+}$$
          Weights severe defaults progressively higher than minor late payment slips.
        
        - **Payment Reliability Score**:
          $$\\text{Reliability} = \\frac{\\text{Credit History (Months)}}{1 + 8 \\times \\text{Delinquency Severity Index}}$$
          Rewards long clean history, significantly discounting profiles with recent defaults.
        
        - **Credit Inquiry Density**:
          $$\\text{Density} = \\frac{\\text{Inquiries (Last 6M)}}{\\text{Open Credit Lines} + 1}$$
          Flags recent aggressive credit-seeking behavior.
        """)

    st.markdown("---")
    st.subheader("Explore Raw & Engineered Credit Dataset")
    if os.path.exists("data/credit_data.csv"):
        data_df = pd.read_csv("data/credit_data.csv")
        st.markdown(f"**Total Records:** {len(data_df):,} applicants &nbsp;|&nbsp; **Variables:** {data_df.shape[1]}")
        st.dataframe(data_df.head(20), use_container_width=True)

        st.markdown("#### Statistical Distribution Summary")
        st.dataframe(data_df.describe(), use_container_width=True)
    else:
        st.info("Run `python train.py` to create the initial dataset.")

# -------------------------------------------------------------
# TAB 4: Batch Applicant Underwriting
# -------------------------------------------------------------
with tab4:
    st.subheader("Batch Loan Applicant Underwriting")
    st.markdown("Upload a CSV file containing applicant profiles, or test using a generated evaluation cohort.")

    c_b1, c_b2 = st.columns([1, 1])
    with c_b1:
        uploaded_file = st.file_uploader("Upload Applicants CSV", type=['csv'])
    with c_b2:
        batch_model = st.selectbox("Underwriting Engine", ["Random Forest", "Gradient Boosting", "Logistic Regression", "Decision Tree"], key="batch_engine")

    if st.button("🚀 Underwrite Evaluation Cohort (50 applicants)"):
        sample_batch = generate_credit_data(n_samples=50, random_state=99)
        pipe_path = f"models/{batch_model.replace(' ', '_')}_pipeline.joblib"
        
        if os.path.exists(pipe_path):
            scored = predict_batch(sample_batch, pipeline_path=pipe_path)
            
            st.success(f"Successfully scored {len(scored)} applicants using {batch_model}!")
            
            # Summary Metrics
            app_rate = (scored['decision'] == 'APPROVED').mean()
            rev_rate = (scored['decision'] == 'MANUAL REVIEW').mean()
            dec_rate = (scored['decision'] == 'DECLINED').mean()
            avg_score = scored['credit_score'].mean()

            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Approval Rate", f"{app_rate:.1%}")
            b2.metric("Manual Review Rate", f"{rev_rate:.1%}")
            b3.metric("Decline Rate", f"{dec_rate:.1%}")
            b4.metric("Avg FICO Score", f"{avg_score:.0f}")

            st.dataframe(
                scored[['credit_score', 'risk_tier', 'decision', 'prob_creditworthy', 'prob_default', 'annual_income', 'loan_amount', 'credit_card_utilization', 'num_late_payments_30_59_days']],
                use_container_width=True
            )

            # Download CSV
            csv_data = scored.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Scored Applicants CSV",
                data=csv_data,
                file_name="underwritten_applicants_scored.csv",
                mime="text/csv"
            )
        else:
            st.error(f"Model file `{pipe_path}` does not exist. Run `python train.py` first.")

    if uploaded_file is not None:
        try:
            up_df = pd.read_csv(uploaded_file)
            pipe_path = f"models/{batch_model.replace(' ', '_')}_pipeline.joblib"
            scored_up = predict_batch(up_df, pipeline_path=pipe_path)
            st.success(f"Underwritten {len(scored_up)} applicants from uploaded file!")
            st.dataframe(scored_up.head(50), use_container_width=True)

            csv_up = scored_up.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Scored Uploaded File",
                data=csv_up,
                file_name="scored_uploaded_batch.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Error processing uploaded CSV: {e}")
