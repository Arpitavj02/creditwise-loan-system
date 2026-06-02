# 🏦 CreditWise Loan System

project url: https://creditwise-loan-system-d4o9g5wregjasf33xdi8hq.streamlit.app/

> **Minor Project** — Machine Learning powered Loan Approval Prediction System  
> Built for SecureTrust Bank using Python, Scikit-learn, XGBoost & Streamlit

---

## 📁 Project Structure

```
creditwise/
├── data/
│   ├── generate_data.py       # Synthetic dataset generator
│   └── loan_data.csv          # Generated dataset (2000 records)
├── models/
│   └── creditwise_artifacts.pkl  # Trained models + encoders (auto-generated)
├── src/
│   └── train_model.py         # Full ML training pipeline
├── app.py                     # Streamlit web application
├── requirements.txt           # Python dependencies
└── README.md
```

---

## ⚙️ Setup in VS Code (Step-by-Step)

### Step 1: Open the Project
```bash
# Open this folder in VS Code
code creditwise/
```

### Step 2: Create a Virtual Environment
```bash
# In VS Code terminal (Ctrl + `)
python -m venv venv

# Activate it:
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Generate the Dataset
```bash
cd data
python generate_data.py
cd ..
```
✅ This creates `data/loan_data.csv` with 2000 applicant records.

### Step 5: Train the Models
```bash
python src/train_model.py
```
✅ This trains 3 models and saves `models/creditwise_artifacts.pkl`.

**Expected output:**
```
[TRAINING] Logistic Regression... Accuracy: 0.82  ROC-AUC: 0.91
[TRAINING] Random Forest...       Accuracy: 0.89  ROC-AUC: 0.95
[TRAINING] XGBoost...             Accuracy: 0.89  ROC-AUC: 0.96
[BEST MODEL] XGBoost with ROC-AUC = 0.96
```

### Step 6: Launch the App
```bash
streamlit run app.py
```
✅ Opens in browser at `http://localhost:8501`

---

## 🎯 Features

| Page | Description |
|------|-------------|
| 🔍 Loan Prediction | Fill applicant form → instant Approved/Rejected result with confidence % |
| 📊 Model Performance | Compare all 3 models with metrics, charts, confusion matrices |
| 📈 Feature Insights | Feature importance bar chart + key factor explanations |
| ℹ️ About | Project overview, architecture, feature list |

---

## 🤖 Models & Results

| Model | Accuracy | ROC-AUC | F1 Score |
|-------|----------|---------|----------|
| Logistic Regression | ~82% | ~91% | ~78% |
| Random Forest | ~89% | ~95% | ~85% |
| **XGBoost** ⭐ | **~89%** | **~96%** | **~84%** |

---

## 📊 Dataset Features

| Feature | Type | Description |
|---------|------|-------------|
| Applicant_Income | Numeric | Monthly income |
| Coapplicant_Income | Numeric | Co-applicant income |
| Credit_Score | Numeric | Bureau score (300–900) |
| DTI_Ratio | Numeric | Debt-to-Income ratio |
| Loan_Amount | Numeric | Requested loan amount |
| Loan_Term | Numeric | Duration in months |
| Employment_Status | Categorical | Salaried/Self-Employed/Business |
| Education_Level | Categorical | Graduate/Postgraduate/Undergraduate |
| ... | ... | (19 features total) |

---

## 🧠 Engineered Features (auto-created during training)

- `Total_Income` = Applicant + Co-applicant income
- `Loan_to_Income_Ratio` = Loan amount / Total income
- `Collateral_to_Loan_Ratio` = Collateral value / Loan amount
- `Savings_to_Loan_Ratio` = Savings / Loan amount
- `Income_Per_Dependent` = Total income / (Dependents + 1)

---

## 💡 Key Findings

1. **Credit Score** is the most predictive feature (28% importance)
2. **DTI Ratio ≤ 0.35** strongly indicates approval
3. **Savings-to-Loan ratio** signals financial discipline
4. **Collateral value** reduces bank risk significantly

---

*Built with Python 3.10+ | scikit-learn | XGBoost | Streamlit*
