

import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, f1_score, precision_score, recall_score
)
from xgboost import XGBClassifier


BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, '..', 'data', 'loan_data.csv')
MODEL_DIR = os.path.join(BASE, '..', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
print(f"[INFO] Dataset loaded: {df.shape}")


df['Total_Income'] = df['Applicant_Income'] + df['Coapplicant_Income']
df['Income_Per_Dependent'] = df['Total_Income'] / (df['Dependents'] + 1)
df['Loan_to_Income_Ratio'] = df['Loan_Amount'] / (df['Total_Income'] + 1)
df['Collateral_to_Loan_Ratio'] = df['Collateral_Value'] / (df['Loan_Amount'] + 1)
df['Savings_to_Loan_Ratio'] = df['Savings'] / (df['Loan_Amount'] + 1)


CAT_COLS = [
    'Employment_Status', 'Marital_Status', 'Loan_Purpose',
    'Property_Area', 'Education_Level', 'Gender', 'Employer_Category'
]

encoders = {}
for col in CAT_COLS:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

DROP_COLS = ['Applicant_ID', 'Loan_Approved']
FEATURE_COLS = [c for c in df.columns if c not in DROP_COLS]

X = df[FEATURE_COLS]
y = df['Loan_Approved']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"[INFO] Train: {X_train.shape}, Test: {X_test.shape}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

l
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    'Random Forest':       RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, class_weight='balanced'),
    'XGBoost':             XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                                         use_label_encoder=False, eval_metric='logloss',
                                         random_state=42, verbosity=0)
}


results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\n[TRAINING] {name}...")

    # Use scaled data for LR
    Xtr = X_train_scaled if name == 'Logistic Regression' else X_train
    Xte = X_test_scaled  if name == 'Logistic Regression' else X_test
    Xcv = X_train_scaled if name == 'Logistic Regression' else X_train

    model.fit(Xtr, y_train)
    preds = model.predict(Xte)
    proba = model.predict_proba(Xte)[:, 1]

    acc  = accuracy_score(y_test, preds)
    auc  = roc_auc_score(y_test, proba)
    f1   = f1_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec  = recall_score(y_test, preds)
    cv_scores = cross_val_score(model, Xcv, y_train, cv=cv, scoring='roc_auc')

    results[name] = {
        'model':      model,
        'accuracy':   acc,
        'roc_auc':    auc,
        'f1_score':   f1,
        'precision':  prec,
        'recall':     rec,
        'cv_mean':    cv_scores.mean(),
        'cv_std':     cv_scores.std(),
        'conf_matrix': confusion_matrix(y_test, preds),
        'report':     classification_report(y_test, preds)
    }

    print(f"  Accuracy : {acc:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  CV AUC   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")


best_name = max(results, key=lambda k: results[k]['roc_auc'])
print(f"\n[BEST MODEL] {best_name} with ROC-AUC = {results[best_name]['roc_auc']:.4f}")


rf_model = results['Random Forest']['model']
feature_importance = pd.DataFrame({
    'Feature': FEATURE_COLS,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)


artifacts = {
    'encoders':           encoders,
    'scaler':             scaler,
    'feature_cols':       FEATURE_COLS,
    'models':             {k: v['model'] for k, v in results.items()},
    'best_model_name':    best_name,
    'metrics':            {k: {mk: mv for mk, mv in v.items() if mk != 'model'} for k, v in results.items()},
    'feature_importance': feature_importance,
}

with open(os.path.join(MODEL_DIR, 'creditwise_artifacts.pkl'), 'wb') as f:
    pickle.dump(artifacts, f)

print(f"\n[SAVED] Artifacts saved to models/creditwise_artifacts.pkl")
print("\n[TOP 10 FEATURES]")
print(feature_importance.head(10).to_string(index=False))
