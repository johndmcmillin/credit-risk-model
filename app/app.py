"""
Credit Risk Scoring Model — Interactive Application
McMillin Analytics

Allows users to input borrower characteristics and receive a predicted
default probability with explanations of key risk drivers.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import os

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Credit Risk Scorer | McMillin Analytics",
    page_icon="📊",
    layout="wide"
)

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    """Load trained models and metadata."""
    app_dir = os.path.dirname(os.path.abspath(__file__))
    xgb_model = joblib.load(os.path.join(app_dir, 'xgb_model.pkl'))
    feature_info = pd.read_csv(os.path.join(app_dir, 'feature_info.csv'))
    model_features = xgb_model.get_booster().feature_names
    return xgb_model, feature_info, model_features

try:
    xgb_model, feature_info, model_features = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    st.warning(f"Models not yet trained. Run notebooks 01-03 first. ({e})")


# --- HEADER ---
st.markdown("""
# 📊 Credit Risk Scoring Model
### McMillin Analytics

Predict the probability of loan default based on borrower characteristics.  
Built with domain-driven feature engineering applied to machine learning.

---
""")

# ============================================================
# SIDEBAR: BORROWER INPUTS
# ============================================================
st.sidebar.header("Borrower Profile")

# --- PROPOSED LOAN ---
st.sidebar.subheader("📝 Proposed Loan")
loan_amount = st.sidebar.number_input(
    "Loan Amount ($)",
    min_value=1000, max_value=40000, value=15000, step=1000
)
annual_income = st.sidebar.number_input(
    "Annual Income ($)",
    min_value=10000, max_value=500000, value=75000, step=5000
)
dti = st.sidebar.slider(
    "Existing DTI — before proposed loan (%)",
    min_value=0.0, max_value=60.0, value=18.0, step=0.5
)
int_rate = st.sidebar.slider(
    "Interest Rate (%)",
    min_value=5.0, max_value=30.0, value=12.0, step=0.25
)
term_months = st.sidebar.selectbox(
    "Loan Term",
    options=[36, 60],
    index=0,
    format_func=lambda x: f"{x} months"
)

# --- CALCULATED METRICS (immediately after loan inputs) ---
installment = loan_amount * (int_rate / 100 / 12) / (1 - (1 + int_rate / 100 / 12) ** (-term_months))
existing_monthly_debt = (dti / 100) * (annual_income / 12)
total_monthly_debt = existing_monthly_debt + installment
dti_pro_forma = (total_monthly_debt / (annual_income / 12)) * 100
payment_to_income = (installment / (annual_income / 12)) * 100
loan_to_income = loan_amount / max(annual_income, 1)

st.sidebar.markdown("---")
st.sidebar.metric("Monthly Payment", f"${installment:,.0f}")
st.sidebar.metric("Pro Forma DTI", f"{dti_pro_forma:.1f}%",
                   delta=f"+{dti_pro_forma - dti:.1f}% from existing",
                   delta_color="inverse")
st.sidebar.metric("Payment-to-Income", f"{payment_to_income:.1f}%")
st.sidebar.markdown("---")

purpose = st.sidebar.selectbox(
    "Loan Purpose",
    options=['debt_consolidation', 'credit_card', 'home_improvement',
             'major_purchase', 'small_business', 'medical', 'car', 'other']
)

# --- CAPACITY ---
st.sidebar.subheader("💰 Capacity")
emp_length = st.sidebar.selectbox(
    "Employment Length",
    options=['< 1 year', '1 year', '2 years', '3 years', '4 years',
             '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years'],
    index=5
)

# --- CHARACTER ---
st.sidebar.subheader("📋 Credit History")
fico_score = st.sidebar.slider(
    "FICO Score",
    min_value=300, max_value=850, value=700, step=5
)
delinq_2yrs = st.sidebar.number_input(
    "Delinquencies (last 2 years)",
    min_value=0, max_value=10, value=0
)
inq_last_6mths = st.sidebar.number_input(
    "Credit Inquiries (last 6 months)",
    min_value=0, max_value=10, value=1
)
pub_rec = st.sidebar.number_input(
    "Public Records",
    min_value=0, max_value=10, value=0
)
pub_rec_bankruptcies = st.sidebar.number_input(
    "Bankruptcies",
    min_value=0, max_value=5, value=0
)
credit_history_years = st.sidebar.slider(
    "Credit History Length (years)",
    min_value=1, max_value=40, value=12
)
open_acc = st.sidebar.number_input(
    "Open Accounts",
    min_value=0, max_value=50, value=10
)
total_acc = st.sidebar.number_input(
    "Total Accounts (lifetime)",
    min_value=1, max_value=100, value=25
)
mort_acc = st.sidebar.number_input(
    "Mortgage Accounts",
    min_value=0, max_value=20, value=1
)

# --- CAPITAL & LEVERAGE ---
st.sidebar.subheader("🏦 Capital & Leverage")
home_ownership = st.sidebar.selectbox(
    "Home Ownership",
    options=['MORTGAGE', 'RENT', 'OWN', 'OTHER']
)
revol_util = st.sidebar.slider(
    "Revolving Utilization (%)",
    min_value=0.0, max_value=120.0, value=45.0, step=1.0
)
revol_bal = st.sidebar.number_input(
    "Revolving Balance ($)",
    min_value=0, max_value=200000, value=12000, step=1000
)
total_rev_hi_lim = st.sidebar.number_input(
    "Total Revolving Credit Limit ($)",
    min_value=0, max_value=500000, value=30000, step=5000
)
tot_cur_bal = st.sidebar.number_input(
    "Total Current Balance — all accounts ($)",
    min_value=0, max_value=1000000, value=80000, step=5000
)


# ============================================================
# DERIVED FEATURES
# ============================================================
revol_bal_to_income = revol_bal / max(annual_income, 1)
total_leverage = tot_cur_bal / max(annual_income, 1)
revol_headroom = total_rev_hi_lim - revol_bal
account_util_rate = open_acc / max(total_acc, 1)
total_derog_marks = delinq_2yrs + pub_rec + pub_rec_bankruptcies
high_inquiry_flag = 1 if inq_last_6mths >= 3 else 0

# Employment length mapping
emp_map = {
    '< 1 year': 0.5, '1 year': 1, '2 years': 2, '3 years': 3,
    '4 years': 4, '5 years': 5, '6 years': 6, '7 years': 7,
    '8 years': 8, '9 years': 9, '10+ years': 10
}
emp_length_num = emp_map[emp_length]

# Rate premium (vs. average rate for typical C-grade loan as baseline)
rate_premium = int_rate - 14.5

# Home ownership dummies
home_RENT = 1 if home_ownership == 'RENT' else 0
home_OWN = 1 if home_ownership == 'OWN' else 0
home_OTHER = 1 if home_ownership == 'OTHER' else 0

# Purpose dummies
purpose_debt_consolidation = 1 if purpose == 'debt_consolidation' else 0
purpose_credit_card = 1 if purpose == 'credit_card' else 0
purpose_home_improvement = 1 if purpose == 'home_improvement' else 0
purpose_major_purchase = 1 if purpose == 'major_purchase' else 0
purpose_small_business = 1 if purpose == 'small_business' else 0
purpose_medical = 1 if purpose == 'medical' else 0
purpose_other = 1 if purpose in ['car', 'other'] else 0

# Is joint (default to individual application)
is_joint = 0

# Derived defaults for features not directly collected
tot_hi_cred_lim = tot_cur_bal * 1.3
total_bc_limit = total_rev_hi_lim * 0.7
num_actv_bc_tl = max(int(open_acc * 0.4), 1)
num_sats = open_acc
pct_tl_nvr_dlq = 95.0 if total_derog_marks == 0 else max(80.0, 95.0 - total_derog_marks * 5)
total_bal_ex_mort = max(tot_cur_bal - (mort_acc * 150000 if mort_acc > 0 else 0), revol_bal)
total_il_high_credit_limit = tot_hi_cred_lim - total_rev_hi_lim
num_bc_tl = max(int(open_acc * 0.3), 1)


# ============================================================
# MAIN CONTENT
# ============================================================
if models_loaded:

    # --- BUILD FEATURE VECTOR (50 features, no grade) ---
    feature_values = {
        'int_rate': int_rate,
        'term_months': term_months,
        'home_RENT': home_RENT,
        'dti_pro_forma': dti_pro_forma,
        'loan_to_income': loan_to_income,
        'mort_acc': mort_acc,
        'purpose_small_business': purpose_small_business,
        'fico_score': fico_score,
        'inq_last_6mths': inq_last_6mths,
        'tot_hi_cred_lim': tot_hi_cred_lim,
        'rate_premium': rate_premium,
        'num_actv_bc_tl': num_actv_bc_tl,
        'emp_length_num': emp_length_num,
        'is_joint': is_joint,
        'home_OWN': home_OWN,
        'dti': dti,
        'total_bc_limit': total_bc_limit,
        'total_leverage': total_leverage,
        'account_util_rate': account_util_rate,
        'pub_rec': pub_rec,
        'loan_amnt': loan_amount,
        'purpose_credit_card': purpose_credit_card,
        'payment_to_income': payment_to_income,
        'total_derog_marks': total_derog_marks,
        'purpose_other': purpose_other,
        'num_sats': num_sats,
        'annual_inc': annual_income,
        'purpose_medical': purpose_medical,
        'revol_bal_to_income': revol_bal_to_income,
        'installment': installment,
        'revol_bal': revol_bal,
        'credit_history_years': credit_history_years,
        'delinq_2yrs': delinq_2yrs,
        'purpose_debt_consolidation': purpose_debt_consolidation,
        'pub_rec_bankruptcies': pub_rec_bankruptcies,
        'tot_cur_bal': tot_cur_bal,
        'total_il_high_credit_limit': total_il_high_credit_limit,
        'open_acc': open_acc,
        'pct_tl_nvr_dlq': pct_tl_nvr_dlq,
        'purpose_home_improvement': purpose_home_improvement,
        'revol_headroom': revol_headroom,
        'purpose_major_purchase': purpose_major_purchase,
        'num_bc_tl': num_bc_tl,
        'revol_util': revol_util,
        'total_bal_ex_mort': total_bal_ex_mort,
        'total_rev_hi_lim': total_rev_hi_lim,
        'total_acc': total_acc,
        'home_OTHER': home_OTHER,
        'high_inquiry_flag': high_inquiry_flag,
    }

    # Build dataframe in exact model feature order
    input_df = pd.DataFrame([feature_values])[model_features]

    # --- PREDICT ---
    risk_score = xgb_model.predict_proba(input_df)[0][1]

    # --- IMPLIED GRADE ---
    # Map predicted default probability back to Lending Club grade equivalent
    grade_thresholds = [
        ('A', 0.07),
        ('B', 0.13),
        ('C', 0.18),
        ('D', 0.24),
        ('E', 0.30),
        ('F', 0.37),
        ('G', 1.00),
    ]
    implied_grade = 'G'
    for g, threshold in grade_thresholds:
        if risk_score <= threshold:
            implied_grade = g
            break

    # --- DISPLAY ---
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Risk Assessment")

        # Risk gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score * 100,
            number={'suffix': '%', 'font': {'size': 48}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 15], 'color': '#2ecc71'},
                    {'range': [15, 30], 'color': '#f1c40f'},
                    {'range': [30, 50], 'color': '#e67e22'},
                    {'range': [50, 100], 'color': '#e74c3c'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': risk_score * 100
                }
            },
            title={'text': "Default Probability", 'font': {'size': 20}}
        ))
        fig.update_layout(height=300, margin=dict(t=50, b=0, l=30, r=30))
        st.plotly_chart(fig, use_container_width=True)

        # Risk rating
        if risk_score < 0.10:
            rating, color = "LOW RISK", "#2ecc71"
        elif risk_score < 0.20:
            rating, color = "MODERATE RISK", "#f1c40f"
        elif risk_score < 0.35:
            rating, color = "ELEVATED RISK", "#e67e22"
        else:
            rating, color = "HIGH RISK", "#e74c3c"

        st.markdown(f"""
        <div style="text-align: center; padding: 10px; background-color: {color}20;
             border-left: 4px solid {color}; border-radius: 4px;">
            <h3 style="color: {color}; margin: 0;">{rating}</h3>
            <p style="margin: 5px 0 0 0; font-size: 1.1em;"><strong>Implied Grade: {implied_grade}</strong></p>
        </div>
        """, unsafe_allow_html=True)

        st.caption("XGBoost model — no Lending Club grade inputs (AUC-ROC: 0.7257)")

    with col2:
        st.subheader("Key Risk Drivers")

        factors = {
            'FICO Score': {'value': fico_score, 'benchmark': 700,
                           'direction': 'lower_is_worse'},
            'Pro Forma DTI': {'value': dti_pro_forma, 'benchmark': 25,
                              'direction': 'higher_is_worse'},
            'Revolving Utilization': {'value': revol_util, 'benchmark': 40,
                                     'direction': 'higher_is_worse'},
            'Interest Rate': {'value': int_rate, 'benchmark': 12,
                              'direction': 'higher_is_worse'},
            'Payment-to-Income': {'value': payment_to_income, 'benchmark': 10,
                                  'direction': 'higher_is_worse'},
            'Loan-to-Income': {'value': loan_to_income * 100, 'benchmark': 30,
                               'direction': 'higher_is_worse'},
            'Delinquencies (2yr)': {'value': float(delinq_2yrs), 'benchmark': 0,
                                    'direction': 'higher_is_worse'},
            'Credit Inquiries (6mo)': {'value': float(inq_last_6mths), 'benchmark': 2,
                                       'direction': 'higher_is_worse'},
            'Credit History': {'value': float(credit_history_years), 'benchmark': 10,
                               'direction': 'lower_is_worse'},
        }

        for name, info in factors.items():
            if info['direction'] == 'higher_is_worse':
                status = "🟢" if info['value'] <= info['benchmark'] else "🔴"
            else:
                status = "🟢" if info['value'] >= info['benchmark'] else "🔴"

            st.markdown(f"""
            **{status} {name}:** {info['value']:.1f}
            *(benchmark: {info['benchmark']})*
            """)

    # --- CREDIT ANALYSIS NARRATIVE ---
    st.markdown("---")
    st.subheader("📝 Credit Analysis Summary")

    strengths = []
    concerns = []

    if fico_score >= 720:
        strengths.append(f"Strong FICO score ({fico_score}) indicates solid credit management history")
    elif fico_score < 660:
        concerns.append(f"Subprime FICO score ({fico_score}) indicates elevated credit risk")

    if dti_pro_forma <= 25:
        strengths.append(f"Pro forma DTI ({dti_pro_forma:.1f}%) indicates adequate capacity — existing {dti:.1f}% plus proposed payment")
    elif dti_pro_forma > 40:
        concerns.append(f"Elevated pro forma DTI ({dti_pro_forma:.1f}%) after proposed loan — up from {dti:.1f}% existing. Borrower may be stretched thin")

    if revol_util <= 30:
        strengths.append(f"Low revolving utilization ({revol_util:.0f}%) indicates available credit headroom")
    elif revol_util > 75:
        concerns.append(f"High revolving utilization ({revol_util:.0f}%) signals potential liquidity stress")

    if delinq_2yrs == 0 and pub_rec == 0 and pub_rec_bankruptcies == 0:
        strengths.append("Clean derogatory history — no delinquencies, public records, or bankruptcies")
    if delinq_2yrs > 0:
        concerns.append(f"{delinq_2yrs} delinquency(ies) in last 2 years — pattern warrants scrutiny")
    if pub_rec_bankruptcies > 0:
        concerns.append(f"{pub_rec_bankruptcies} bankruptcy(ies) on record — significant credit event")

    if purpose == 'small_business':
        concerns.append("Small business purpose — using personal credit for business needs increases risk")

    if term_months == 60:
        concerns.append("60-month term carries higher risk than 36-month — longer exposure period")

    if payment_to_income > 15:
        concerns.append(f"Monthly payment (${installment:,.0f}) represents {payment_to_income:.1f}% of monthly income")

    if credit_history_years < 5:
        concerns.append(f"Limited credit history ({credit_history_years} years) — thin file increases uncertainty")
    elif credit_history_years >= 15:
        strengths.append(f"Well-established credit history ({credit_history_years} years)")

    if loan_to_income > 0.5:
        concerns.append(f"Loan amount represents {loan_to_income*100:.0f}% of annual income")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Strengths:**")
        if strengths:
            for s in strengths:
                st.markdown(f"- ✅ {s}")
        else:
            st.markdown("- *No notable strengths identified*")

    with col2:
        st.markdown("**Concerns:**")
        if concerns:
            for c in concerns:
                st.markdown(f"- ⚠️ {c}")
        else:
            st.markdown("- *No notable concerns identified*")

else:
    st.markdown("""
    ## 🚀 Getting Started

    To use this app, you'll need to train the models first:

    1. Download the [Lending Club dataset from Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
    2. Place `accepted_2007_to_2018Q4.csv` in the `data/raw/` directory
    3. Run the notebooks in order:
        - `01_data_exploration.ipynb`
        - `02_feature_engineering.ipynb`
        - `03_modeling.ipynb`
    4. The trained models will be saved to the `app/` directory
    5. Restart this Streamlit app
    """)


# --- FOOTER ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.9em;">
    McMillin Analytics | Credit Risk Scoring Model<br>
    Built with Python, scikit-learn, XGBoost, and Streamlit
</div>
""", unsafe_allow_html=True)
