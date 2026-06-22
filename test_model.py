"""
Test script to verify model predictions with saved label encoders
"""
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load model, scaler, columns, and label encoders
model = joblib.load('models/loan_model.pkl')
scaler = joblib.load('models/scaler.pkl')
columns = joblib.load('models/columns.pkl')
label_encoders = joblib.load('models/label_encoders.pkl')

print(f"Model type: {type(model)}")
print(f"Model features: {model.n_features_in_}")
print(f"Columns: {columns}")
print(f"Label encoders available for: {list(label_encoders.keys())}")
print()

# Test Case 1: Good applicant (should be approved)
print("=" * 50)
print("Test Case 1: Good Applicant (Should be Approved)")
print("=" * 50)
good_data = {
    'Gender': ['Male'],
    'Married': ['Yes'],
    'Dependents': ['0'],
    'Education': ['Graduate'],
    'Self_Employed': ['No'],
    'ApplicantIncome': [50000],
    'CoapplicantIncome': [20000],
    'LoanAmount': [300000],
    'Loan_Amount_Term': [360],
    'Credit_History': [1],
    'Property_Area': ['Urban']
}

good_df = pd.DataFrame(good_data)
good_df['TotalIncome'] = good_df['ApplicantIncome'] + good_df['CoapplicantIncome']
good_df['Income_Loan_Ratio'] = good_df['TotalIncome'] / (good_df['LoanAmount'] + 1)
good_df['EMI'] = good_df['LoanAmount'] / good_df['Loan_Amount_Term']
good_df['RiskScore'] = good_df['LoanAmount'] / (good_df['TotalIncome'] + 1)

print(f"Engineered features:")
print(f"TotalIncome: {good_df['TotalIncome'].values[0]}")
print(f"Income_Loan_Ratio: {good_df['Income_Loan_Ratio'].values[0]:.4f}")
print(f"EMI: {good_df['EMI'].values[0]:.2f}")
print(f"RiskScore: {good_df['RiskScore'].values[0]:.4f}")

# Encode categorical variables using saved label encoders
for col in good_df.select_dtypes(include='object').columns:
    if col in label_encoders:
        good_df[col] = label_encoders[col].transform(good_df[col])
        print(f"Encoded {col} using saved LabelEncoder")
    else:
        print(f"Warning: No LabelEncoder for {col}")

# Reindex to match training columns
good_df = good_df.reindex(columns=columns, fill_value=0)

# Scale
good_array = scaler.transform(good_df)

# Predict with adjusted threshold
prediction_proba = model.predict_proba(good_array)[0]

# Use a lower threshold for approval (0.35 instead of 0.5) to be less conservative
approval_threshold = 0.35
if prediction_proba[1] >= approval_threshold:
    prediction = 1
else:
    prediction = 0

confidence = max(prediction_proba) * 100

print(f"Prediction: {prediction}")
print(f"Probabilities: {prediction_proba}")
print(f"Approval probability: {prediction_proba[1]:.4f}")
print(f"Confidence: {confidence:.2f}%")
print(f"Status: {'Approved' if prediction == 1 else 'Rejected'}")
print()

# Test Case 2: Bad applicant (should be rejected)
print("=" * 50)
print("Test Case 2: Bad Applicant (Should be Rejected)")
print("=" * 50)
bad_data = {
    'Gender': ['Male'],
    'Married': ['No'],
    'Dependents': ['3+'],
    'Education': ['Not Graduate'],
    'Self_Employed': ['Yes'],
    'ApplicantIncome': [10000],
    'CoapplicantIncome': [0],
    'LoanAmount': [900000],
    'Loan_Amount_Term': [12],
    'Credit_History': [0],
    'Property_Area': ['Rural']
}

bad_df = pd.DataFrame(bad_data)
bad_df['TotalIncome'] = bad_df['ApplicantIncome'] + bad_df['CoapplicantIncome']
bad_df['Income_Loan_Ratio'] = bad_df['TotalIncome'] / (bad_df['LoanAmount'] + 1)
bad_df['EMI'] = bad_df['LoanAmount'] / bad_df['Loan_Amount_Term']
bad_df['RiskScore'] = bad_df['LoanAmount'] / (bad_df['TotalIncome'] + 1)

print(f"Engineered features:")
print(f"TotalIncome: {bad_df['TotalIncome'].values[0]}")
print(f"Income_Loan_Ratio: {bad_df['Income_Loan_Ratio'].values[0]:.4f}")
print(f"EMI: {bad_df['EMI'].values[0]:.2f}")
print(f"RiskScore: {bad_df['RiskScore'].values[0]:.4f}")

# Encode categorical variables using saved label encoders
for col in bad_df.select_dtypes(include='object').columns:
    if col in label_encoders:
        bad_df[col] = label_encoders[col].transform(bad_df[col])
        print(f"Encoded {col} using saved LabelEncoder")
    else:
        print(f"Warning: No LabelEncoder for {col}")

# Reindex to match training columns
bad_df = bad_df.reindex(columns=columns, fill_value=0)

# Scale
bad_array = scaler.transform(bad_df)

# Predict with adjusted threshold
prediction_proba = model.predict_proba(bad_array)[0]

# Use a lower threshold for approval (0.35 instead of 0.5) to be less conservative
approval_threshold = 0.35
if prediction_proba[1] >= approval_threshold:
    prediction = 1
else:
    prediction = 0

confidence = max(prediction_proba) * 100

print(f"Prediction: {prediction}")
print(f"Probabilities: {prediction_proba}")
print(f"Approval probability: {prediction_proba[1]:.4f}")
print(f"Confidence: {confidence:.2f}%")
print(f"Status: {'Approved' if prediction == 1 else 'Rejected'}")
