"""
CreditWise Loan System - Dataset Generator
Generates a realistic synthetic loan dataset matching the problem statement schema.
"""

import pandas as pd
import numpy as np

np.random.seed(42)
N = 2000

def generate_dataset(n=N):
    # --- Basic Demographics ---
    age = np.random.randint(21, 65, n)
    gender = np.random.choice(['Male', 'Female'], n, p=[0.65, 0.35])
    marital_status = np.random.choice(['Married', 'Single'], n, p=[0.60, 0.40])
    dependents = np.random.choice([0, 1, 2, 3, 4], n, p=[0.30, 0.25, 0.25, 0.15, 0.05])
    education_level = np.random.choice(['Graduate', 'Postgraduate', 'Undergraduate'], n, p=[0.50, 0.25, 0.25])
    property_area = np.random.choice(['Urban', 'Semi-Urban', 'Rural'], n, p=[0.40, 0.35, 0.25])

    # --- Employment & Income ---
    employment_status = np.random.choice(['Salaried', 'Self-Employed', 'Business'], n, p=[0.55, 0.25, 0.20])
    employer_category = np.random.choice(['Govt', 'Private', 'Self'], n, p=[0.25, 0.50, 0.25])

    applicant_income = np.where(
        employment_status == 'Salaried',
        np.random.randint(25000, 150000, n),
        np.where(employment_status == 'Business',
                 np.random.randint(40000, 300000, n),
                 np.random.randint(15000, 120000, n))
    )
    coapplicant_income = np.where(
        marital_status == 'Married',
        np.random.randint(0, 80000, n),
        np.zeros(n, dtype=int)
    )

    # --- Financial Info ---
    credit_score = np.clip(np.random.normal(680, 80, n).astype(int), 300, 900)
    existing_loans = np.random.choice([0, 1, 2, 3, 4], n, p=[0.35, 0.30, 0.20, 0.10, 0.05])
    savings = np.random.randint(5000, 500000, n)
    collateral_value = np.random.randint(0, 2000000, n)

    loan_amount = np.random.randint(50000, 5000000, n)
    loan_term = np.random.choice([12, 24, 36, 60, 84, 120, 180, 240, 360], n)
    loan_purpose = np.random.choice(['Home', 'Education', 'Personal', 'Business'], n, p=[0.35, 0.20, 0.25, 0.20])

    total_income = applicant_income + coapplicant_income
    monthly_obligation = (loan_amount / loan_term) * 1.08
    dti_ratio = np.round(np.clip(monthly_obligation / (total_income / 12), 0.05, 1.5), 3)

    # --- Target: Loan Approved (rule-based with noise) ---
    score = np.zeros(n)
    score += (credit_score >= 700) * 2.0
    score += (credit_score >= 750) * 1.5
    score += (dti_ratio <= 0.35) * 2.0
    score += (dti_ratio <= 0.50) * 1.0
    score += (total_income >= 50000) * 1.5
    score += (existing_loans == 0) * 1.0
    score += (existing_loans <= 1) * 0.5
    score += (collateral_value > loan_amount * 0.8) * 1.5
    score += (savings > loan_amount * 0.10) * 1.0
    score += (employment_status == 'Salaried') * 0.5
    score += (education_level == 'Postgraduate') * 0.3
    score += (employer_category == 'Govt') * 0.3
    score -= (dependents >= 3) * 0.5
    score -= (existing_loans >= 3) * 1.0

    noise = np.random.normal(0, 0.8, n)
    prob = 1 / (1 + np.exp(-(score - 5 + noise)))
    loan_approved = (prob > 0.5).astype(int)

    df = pd.DataFrame({
        'Applicant_ID': [f'APP{str(i).zfill(5)}' for i in range(1, n+1)],
        'Applicant_Income': applicant_income,
        'Coapplicant_Income': coapplicant_income,
        'Employment_Status': employment_status,
        'Age': age,
        'Marital_Status': marital_status,
        'Dependents': dependents,
        'Credit_Score': credit_score,
        'Existing_Loans': existing_loans,
        'DTI_Ratio': dti_ratio,
        'Savings': savings,
        'Collateral_Value': collateral_value,
        'Loan_Amount': loan_amount,
        'Loan_Term': loan_term,
        'Loan_Purpose': loan_purpose,
        'Property_Area': property_area,
        'Education_Level': education_level,
        'Gender': gender,
        'Employer_Category': employer_category,
        'Loan_Approved': loan_approved
    })

    return df

if __name__ == '__main__':
    df = generate_dataset()
    df.to_csv('loan_data.csv', index=False)
    print(f"Dataset generated: {df.shape}")
    print(f"Approval rate: {df['Loan_Approved'].mean():.2%}")
    print(df.head())
