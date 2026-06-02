

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path


st.set_page_config(
    page_title="CreditWise Loan System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');
    
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'DM Serif Display', serif; }

    .main { background: #f8f6f1; }
    
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        border-left: 4px solid #1a3c5e;
        margin-bottom: 10px;
    }
    .approved-banner {
        background: linear-gradient(135deg, #1a6b3a, #2d9651);
        color: white;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 600;
        margin: 20px 0;
    }
    .rejected-banner {
        background: linear-gradient(135deg, #8b1a1a, #c0392b);
        color: white;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 600;
        margin: 20px 0;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a3c5e;
        border-bottom: 2px solid #e8e0d0;
        padding-bottom: 6px;
        margin: 20px 0 12px 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    base = Path(__file__).parent
    path = base / 'models' / 'creditwise_artifacts.pkl'
    with open(path, 'rb') as f:
        return pickle.load(f)

artifacts = load_artifacts()
encoders         = artifacts['encoders']
scaler           = artifacts['scaler']
feature_cols     = artifacts['feature_cols']
trained_models   = artifacts['models']
best_model_name  = artifacts['best_model_name']
metrics          = artifacts['metrics']
feat_importance  = artifacts['feature_importance']


st.sidebar.markdown("## CreditWise")
# st.sidebar.markdown("*SecureTrust Bank — ML Loan System*")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🔍 Loan Prediction", "📊 Model Performance", "📈 Feature Insights", "ℹ️ About"]
)


if page == "🔍 Loan Prediction":
    st.title(" CreditWise Loan Prediction")
    st.markdown("Fill in the applicant details below to get an instant loan approval prediction.")

    # Model selector
    model_choice = st.sidebar.selectbox(
        "Select Model",
        list(trained_models.keys()),
        index=list(trained_models.keys()).index(best_model_name)
    )
    st.sidebar.info(f"⭐ Best model: **{best_model_name}**")

    st.markdown('<div class="section-header">👤 Personal Details</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    age              = c1.number_input("Age", 21, 65, 35)
    gender           = c2.selectbox("Gender", ['Male', 'Female'])
    marital_status   = c3.selectbox("Marital Status", ['Married', 'Single'])
    dependents       = c1.slider("Dependents", 0, 4, 1)
    education_level  = c2.selectbox("Education Level", ['Graduate', 'Postgraduate', 'Undergraduate'])
    property_area    = c3.selectbox("Property Area", ['Urban', 'Semi-Urban', 'Rural'])

    st.markdown('<div class="section-header">💼 Employment Details</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    employment_status  = c1.selectbox("Employment Status", ['Salaried', 'Self-Employed', 'Business'])
    employer_category  = c2.selectbox("Employer Category", ['Govt', 'Private', 'Self'])
    applicant_income   = c3.number_input("Monthly Income (₹)", 5000, 1000000, 50000, step=1000)
    coapplicant_income = c1.number_input("Co-applicant Income (₹)", 0, 500000, 0, step=1000)

    st.markdown('<div class="section-header">🏦 Financial Profile</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    credit_score     = c1.slider("Credit Score", 300, 900, 680)
    existing_loans   = c2.slider("Existing Loans", 0, 5, 0)
    savings          = c3.number_input("Savings (₹)", 0, 5000000, 100000, step=5000)
    collateral_value = c1.number_input("Collateral Value (₹)", 0, 10000000, 500000, step=10000)
    dti_ratio        = c2.number_input("DTI Ratio", 0.01, 1.50, 0.35, step=0.01, format="%.2f")

    st.markdown('<div class="section-header">📋 Loan Details</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    loan_amount  = c1.number_input("Loan Amount (₹)", 10000, 10000000, 500000, step=10000)
    loan_term    = c2.selectbox("Loan Term (months)", [12, 24, 36, 60, 84, 120, 180, 240, 360], index=4)
    loan_purpose = c3.selectbox("Loan Purpose", ['Home', 'Education', 'Personal', 'Business'])

    if st.button("🔮 Predict Loan Approval", use_container_width=True, type="primary"):
  
        raw = {
            'Applicant_Income':    applicant_income,
            'Coapplicant_Income':  coapplicant_income,
            'Employment_Status':   employment_status,
            'Age':                 age,
            'Marital_Status':      marital_status,
            'Dependents':          dependents,
            'Credit_Score':        credit_score,
            'Existing_Loans':      existing_loans,
            'DTI_Ratio':           dti_ratio,
            'Savings':             savings,
            'Collateral_Value':    collateral_value,
            'Loan_Amount':         loan_amount,
            'Loan_Term':           loan_term,
            'Loan_Purpose':        loan_purpose,
            'Property_Area':       property_area,
            'Education_Level':     education_level,
            'Gender':              gender,
            'Employer_Category':   employer_category,
        }
        df_row = pd.DataFrame([raw])

  
        total_income = applicant_income + coapplicant_income
        df_row['Total_Income']             = total_income
        df_row['Income_Per_Dependent']     = total_income / (dependents + 1)
        df_row['Loan_to_Income_Ratio']     = loan_amount  / (total_income + 1)
        df_row['Collateral_to_Loan_Ratio'] = collateral_value / (loan_amount + 1)
        df_row['Savings_to_Loan_Ratio']    = savings / (loan_amount + 1)


        cat_cols = ['Employment_Status','Marital_Status','Loan_Purpose',
                    'Property_Area','Education_Level','Gender','Employer_Category']
        for col in cat_cols:
            df_row[col] = encoders[col].transform(df_row[col])

        X_input = df_row[feature_cols]

        model = trained_models[model_choice]
        if model_choice == 'Logistic Regression':
            X_pred = scaler.transform(X_input)
        else:
            X_pred = X_input

        prediction = model.predict(X_pred)[0]
        probability = model.predict_proba(X_pred)[0]

       
        st.markdown("---")
        if prediction == 1:
            st.markdown(
                f'<div class="approved-banner">✅ LOAN APPROVED<br>'
                f'<span style="font-size:1rem;font-weight:400">Confidence: {probability[1]*100:.1f}%</span></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="rejected-banner">❌ LOAN REJECTED<br>'
                f'<span style="font-size:1rem;font-weight:400">Confidence: {probability[0]*100:.1f}%</span></div>',
                unsafe_allow_html=True
            )


        r1, r2, r3 = st.columns([1, 2, 1])
        with r2:
            fig, ax = plt.subplots(figsize=(5, 0.6))
            fig.patch.set_alpha(0)
            ax.set_facecolor('#f8f6f1')
            ax.barh(['Approval Probability'], [probability[1]], color='#2d9651', height=0.5)
            ax.barh(['Approval Probability'], [probability[0]], left=probability[1], color='#c0392b', height=0.5)
            ax.set_xlim(0, 1)
            ax.axvline(0.5, color='white', linestyle='--', linewidth=1.5)
            ax.set_xlabel("Probability")
            for spine in ax.spines.values(): spine.set_visible(False)
            st.pyplot(fig, use_container_width=True)
            plt.close()

        st.markdown("**Key Factors Considered:**")
        kf1, kf2, kf3 = st.columns(3)
        kf1.metric("Credit Score", f"{credit_score}", delta="Good" if credit_score >= 700 else "Low", delta_color="normal" if credit_score >= 700 else "inverse")
        kf2.metric("DTI Ratio", f"{dti_ratio:.2f}", delta="OK" if dti_ratio <= 0.40 else "High", delta_color="normal" if dti_ratio <= 0.40 else "inverse")
        kf3.metric("Total Income", f"₹{total_income:,}")


elif page == "📊 Model Performance":
    st.title("📊 Model Performance Comparison")

    # Metrics table
    rows = []
    for name, m in metrics.items():
        rows.append({
            'Model': name,
            'Accuracy': f"{m['accuracy']:.4f}",
            'ROC-AUC': f"{m['roc_auc']:.4f}",
            'F1 Score': f"{m['f1_score']:.4f}",
            'Precision': f"{m['precision']:.4f}",
            'Recall': f"{m['recall']:.4f}",
            'CV AUC': f"{m['cv_mean']:.4f} ± {m['cv_std']:.4f}",
        })
    df_metrics = pd.DataFrame(rows)
    st.dataframe(df_metrics.set_index('Model'), use_container_width=True)

    # st.info(f"**Best Model:** {best_model_name} (highest ROC-AUC)")

    # Bar chart comparison
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.patch.set_facecolor('#f8f6f1')
    model_names = list(metrics.keys())
    colors = ['#4a90d9', '#2d9651', '#e67e22']
    short = ['LR', 'RF', 'XGB']

    for ax, metric_key, title in zip(axes,
        ['accuracy', 'roc_auc', 'f1_score'],
        ['Accuracy', 'ROC-AUC', 'F1 Score']):

        vals = [metrics[m][metric_key] for m in model_names]
        bars = ax.bar(short, vals, color=colors, edgecolor='white', width=0.5)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylim(0.7, 1.0)
        ax.set_facecolor('#f8f6f1')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='600')

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # Confusion matrices
    st.subheader("Confusion Matrices")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.patch.set_facecolor('#f8f6f1')

    for ax, (name, m) in zip(axes, metrics.items()):
        cm = m['conf_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Rejected','Approved'],
                    yticklabels=['Rejected','Approved'],
                    linewidths=0.5, cbar=False)
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()


elif page == "📈 Feature Insights":
    st.title("📈 Feature Importance & Insights")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("Feature Importance (Random Forest)")
        top15 = feat_importance.head(15)

        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor('#f8f6f1')
        ax.set_facecolor('#f8f6f1')
        colors_bar = ['#1a3c5e' if i < 3 else '#4a90d9' if i < 7 else '#a0bcd6'
                      for i in range(len(top15))]
        bars = ax.barh(top15['Feature'][::-1], top15['Importance'][::-1],
                       color=colors_bar[::-1], edgecolor='white')
        ax.set_xlabel('Importance Score', fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        st.subheader("Top Insights")
        insights = [
            (" Credit Score", "Single most important factor. Scores ≥ 700 strongly predict approval."),
            (" Savings Ratio", "Higher savings relative to loan amount signals financial discipline."),
            (" DTI Ratio", "Debt-to-Income ratio below 0.35 is the sweet spot for approval."),
            (" Collateral", "Strong collateral reduces bank risk significantly."),
            (" Existing Loans", "Multiple running loans increase rejection probability."),
        ]
        for title, desc in insights:
            st.markdown(f"""
            <div class="metric-card">
                <strong>{title}</strong><br>
                <span style="font-size:0.88rem;color:#555">{desc}</span>
            </div>""", unsafe_allow_html=True)

    # Feature table
    st.subheader("All Feature Importances")
    st.dataframe(feat_importance.reset_index(drop=True), use_container_width=True, height=400)

elif page == "ℹ️ About":
    st.title("ℹ️ About CreditWise")
    st.markdown("""
    ### Project Overview
    **CreditWise** is an intelligent loan approval system built for **SecureTrust Bank**
    using Machine Learning to automate and de-bias the loan evaluation process.

    ### Problem Solved
    | Challenge | Solution |
    |-----------|----------|
    | Manual, biased reviews | ML-powered objective scoring |
    | Good customers rejected | Learns complex approval patterns |
    | High-risk customers approved | Trained on historical risk signals |
    | Slow processing | Instant predictions |

    ### Models Used
    | Model | Strengths |
    |-------|-----------|
    | Logistic Regression | Interpretable, fast, good baseline |
    | Random Forest | Handles non-linearity, robust, feature importance |
    | XGBoost | Best accuracy, handles imbalanced data |

    ### Dataset Features (19 input features)
    - **Personal**: Age, Gender, Marital Status, Dependents, Education, Property Area
    - **Employment**: Status, Employer Category, Income
    - **Financial**: Credit Score, DTI Ratio, Savings, Collateral, Existing Loans
    - **Loan**: Amount, Term, Purpose

    ### Engineered Features
    - `Total_Income` = Applicant + Co-applicant Income
    - `Loan_to_Income_Ratio` = Loan Amount / Total Income
    - `Collateral_to_Loan_Ratio` = Collateral / Loan Amount
    - `Savings_to_Loan_Ratio` = Savings / Loan Amount
    - `Income_Per_Dependent` = Total Income / (Dependents + 1)

  
    """)
