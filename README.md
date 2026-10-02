# 💳 CreditIQ - Intelligent Credit Scoring & Underwriting System

[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live%20Demo-brightgreen?style=flat&logo=github)](https://vinila-velmala.github.io/-Credit-Scoring-Model-/)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=vinila-velmala/-Credit-Scoring-Model-&branch=main&mainModule=app.py)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io/)

An end-to-end Machine Learning credit risk assessment platform that predicts an individual's creditworthiness and probability of loan default using historical financial data, domain feature engineering, and classification algorithms.

---

## 🌐 Live Application & Hosting

### 🌍 1. GitHub Pages (Live Instant Web App)
You can directly open the live web dashboard hosted on GitHub Pages:

👉 **[Launch CreditIQ on GitHub Pages](https://vinila-velmala.github.io/-Credit-Scoring-Model-/)**

* **Live URL:** `https://vinila-velmala.github.io/-Credit-Scoring-Model-/`
* **Features:** Instant browser-based credit scoring simulator, dynamic FICO score gauge, multi-model benchmarks, and batch underwriting.

#### How to Enable GitHub Pages in your Repository:
1. Go to your repository on GitHub: **[vinila-velmala/-Credit-Scoring-Model-](https://github.com/vinila-velmala/-Credit-Scoring-Model-)**
2. Click on **Settings** (top bar) > **Pages** (left navigation menu).
3. Under **Build and deployment**:
   * **Source**: Select `Deploy from a branch` (or `GitHub Actions`).
   * **Branch**: Select `main` and folder `/ (root)` or `/docs`, then click **Save**.
4. Your website is instantly live at **`https://vinila-velmala.github.io/-Credit-Scoring-Model-/`**!

---

### 🚀 2. Streamlit Community Cloud (1-Click Deployment)
You can also deploy the full Python Streamlit app to Streamlit Cloud for free:

[![Deploy on Streamlit Community Cloud](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=vinila-velmala/-Credit-Scoring-Model-&branch=main&mainModule=app.py)

**Direct Deployment Link:**  
👉 **[Deploy to Streamlit Community Cloud](https://share.streamlit.io/deploy?repository=vinila-velmala/-Credit-Scoring-Model-&branch=main&mainModule=app.py)**

* **Repository:** `vinila-velmala/-Credit-Scoring-Model-`
* **Branch:** `main`
* **Main file path:** `app.py`

---

### 💻 3. Local Workstation Host Link
When running locally on your computer:
* **Local URL:** [http://localhost:8501](http://localhost:8501)
* **Command to run:**
  ```bash
  streamlit run app.py
  # or
  python -m streamlit run app.py
  ```

---

## 📌 Project Overview & Objectives

* **Core Problem:** Financial institutions require accurate, interpretable, and calibrated risk assessments before granting personal and commercial credit lines to mitigate non-performing loans (NPLs).
* **Objective:** Predict whether an applicant is creditworthy ($y=1$) or at high risk of loan default ($y=0$) using financial history, income, existing debt obligations, and credit utilization.
* **Classification Algorithms Evaluated:**
  * **Logistic Regression** (with balanced class weights and L2 regularization)
  * **Decision Tree Classifier** (interpretable tree-based decision nodes)
  * **Random Forest Classifier** (ensemble bagged decision trees)
  * **Gradient Boosting Classifier** (sequentially boosted decision trees)
* **Key Features:**
  * **Feature Engineering:** Automated derivation of financial ratios (Debt-to-Income, Loan-to-Income, Delinquency Severity Index, Payment Reliability Score, Credit Inquiry Density).
  * **Model Accuracy Assessment:** Precision, Recall, Specificity, F1-Score, ROC-AUC, PR-AUC, Brier Score Loss, and Confusion Matrices.
  * **FICO Score Mapping:** Non-linear calibration converting model probabilities into standard 300–850 credit scores with underwriting tiers (*Excellent*, *Good*, *Fair*, *Poor*).
  * **Interactive Web Dashboard:** Streamlit application with real-time applicant underwriting simulation, model benchmarking visualizations, and batch CSV processing.

---

## 📐 Domain Feature Engineering

In financial credit assessment, nominal dollar values (such as income or total debt) fail to convey liquidity strain without relative scaling. The system computes the following domain-specific financial indicators:

### 1. Debt-To-Income (DTI) Ratio
$$\text{DTI} = \frac{\text{Monthly Debt Payments} + \frac{\text{Loan Amount}}{\text{Loan Term (Months)}}}{\frac{\text{Annual Income}}{12}}$$
*Measures what percentage of monthly gross income is committed to existing debt service plus the proposed loan installment. Traditional lending guidelines flag DTIs above 43%.*

### 2. Loan-To-Income (LTI) Ratio
$$\text{LTI} = \frac{\text{Loan Amount}}{\text{Annual Income}}$$
*Quantifies the scale of principal credit exposure relative to annual earnings capacity.*

### 3. Monthly Disposable Income
$$\text{Disposable} = \frac{\text{Annual Income}}{12} - \left(\text{Monthly Debt Payments} + \frac{\text{Loan Amount}}{\text{Loan Term}}\right)$$
*Measures the applicant's liquid buffer to absorb unforeseen financial emergencies.*

### 4. Delinquency Severity Index
$$\text{Severity} = 1.0 \times \text{Late}_{30-59\text{ days}} + 2.5 \times \text{Late}_{60-89\text{ days}} + 5.0 \times \text{Late}_{90+\text{ days}}$$
*Applies progressive, non-linear penalties to severe delinquency events over minor 30-day slips.*

### 5. Payment Reliability Score
$$\text{Reliability} = \frac{\text{Credit History Length (Months)}}{1 + 8.0 \times \text{Delinquency Severity Index}}$$
*Rewards extensive clean credit track records while discounting profiles with recent defaults.*

### 6. Credit Inquiry Density
$$\text{Inquiry Density} = \frac{\text{Recent Inquiries (Last 6 Months)}}{\text{Open Credit Lines} + 1}$$
*Flags aggressive credit-seeking behavior that often precedes personal bankruptcy.*

---

## 📊 Model Accuracy Assessment & Benchmark Results

Evaluated on a held-out test partition of **2,000 applicants** using stratified cross-validation:

| Classification Algorithm | Accuracy | Precision | Recall | Specificity | F1-Score | ROC-AUC | PR-AUC | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 73.20% | **93.78%** | 73.71% | **70.11%** | 0.8254 | **0.7953** | **0.9504** | 0.1815 |
| **Decision Tree** | 72.95% | 91.60% | 75.45% | 57.65% | 0.8274 | 0.7347 | 0.9285 | 0.1962 |
| **Random Forest** | 81.95% | 91.20% | 87.43% | 48.40% | 0.8928 | 0.7835 | 0.9492 | 0.1374 |
| **Gradient Boosting** | **87.65%** | 88.94% | **97.79%** | 25.62% | **0.9316** | 0.7770 | 0.9470 | **0.0997** |

### Underwriting Insights & Metric Trade-Offs:
* **Precision (93.78% on Logistic Regression):** Crucial in banking and peer-to-peer lending where false positives (approving an applicant who defaults) incur direct capital charge-offs. High precision ensures minimal non-performing loans.
* **ROC-AUC (0.7953 on Logistic Regression):** Demonstrates superior ranking ability and discrimination across diverse policy decision thresholds.
* **Recall & Accuracy (97.79% / 87.65% on Gradient Boosting):** Accurately captures creditworthy applicants, minimizing opportunity costs from rejected good borrowers.

---

## 📈 Visual Performance Analytics

All visual evaluation reports are automatically generated and saved in `reports/`:

* **Multi-Model ROC Curves:** [`reports/roc_curves.png`](reports/roc_curves.png)
* **Precision-Recall Curves:** [`reports/precision_recall_curves.png`](reports/precision_recall_curves.png)
* **Confusion Matrix Grid:** [`reports/confusion_matrices.png`](reports/confusion_matrices.png)
* **Metrics Comparison Bar Chart:** [`reports/metrics_comparison.png`](reports/metrics_comparison.png)
* **Feature Importance (Random Forest):** [`reports/feature_importance_Random_Forest.png`](reports/feature_importance_Random_Forest.png)
* **Feature Weights (Logistic Regression):** [`reports/feature_importance_Logistic_Regression.png`](reports/feature_importance_Logistic_Regression.png)

---

## 🏗️ Project Architecture

```
Credit Scoring Model/
│
├── app.py                      # Interactive Streamlit Web Dashboard
├── train.py                    # End-to-end model training & evaluation pipeline
├── requirements.txt            # Python package dependencies
├── README.md                   # Comprehensive project documentation
├── .gitignore                  # Git ignore rules
│
├── src/
│   ├── dataset.py              # Financial dataset generation & CSV loader
│   ├── features.py             # Domain financial feature engineering pipeline
│   ├── models.py               # Classification models, pipelines & feature importances
│   ├── evaluate.py             # Evaluation metrics, ROC-AUC, PR-AUC & plotting
│   └── predict.py              # Individual & batch scoring, FICO mapping, explainability
│
├── data/
│   └── credit_data.csv         # Financial dataset (10,000 applicant records)
│
├── models/
│   ├── Logistic_Regression_pipeline.joblib
│   ├── Decision_Tree_pipeline.joblib
│   ├── Random_Forest_pipeline.joblib
│   └── Gradient_Boosting_pipeline.joblib
│
├── reports/
│   ├── metrics_summary.csv     # Model evaluation benchmark table
│   ├── metrics_summary.json    # Machine-readable metric results
│   ├── roc_curves.png          # ROC comparison curve
│   ├── precision_recall_curves.png # PR comparison curve
│   ├── confusion_matrices.png  # 2x2 confusion matrix grid
│   ├── metrics_comparison.png  # Grouped metric comparison chart
│   └── feature_importance_*.png # Top risk drivers per model
│
└── tests/
    └── test_pipeline.py        # Automated unit test suite
```

---

## 🚀 Installation & Local Usage

### 1. Clone the Repository
```bash
git clone https://github.com/vinila-velmala/-Credit-Scoring-Model-.git
cd -Credit-Scoring-Model-
```

### 2. Create Virtual Environment & Install Dependencies
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Run the Training Pipeline
Trains all classification algorithms, calculates performance metrics, and generates diagnostic charts:
```bash
python train.py
```

### 4. Run Automated Unit Tests
```bash
python tests/test_pipeline.py
```

### 5. Launch the Web Application
```bash
streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

---

## 🎯 Dashboard Features & Usage Guide

1. **Interactive Underwriting Simulator:**
   - Select quick profile presets (*Prime Borrower*, *Near-Prime Borrower*, *Subprime Borrower*) or customize parameters with sliders.
   - Outputs:
     - Real-time **Credit Score (300–850)**
     - Underwriting Decision: **APPROVED** (Green), **MANUAL REVIEW** (Amber), or **DECLINED** (Red)
     - Key Positive Factors & Identified Risk Flags
     - Calculated Financial Ratios (DTI, LTI, Disposable Income)
2. **Model Accuracy & Benchmarking:**
   - Multi-model comparison table highlighting highest metrics.
   - Interactive display of ROC curves, PR curves, and Confusion Matrices.
3. **Financial History & Feature Engineering Deep-Dive:**
   - Complete mathematical breakdown of credit ratios.
   - Interactive dataset explorer and summary statistics.
4. **Batch Applicant Underwriting:**
   - Upload any CSV file of applicants or run a 50-applicant test cohort with 1-click downloadable scored outputs.

---

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.