"""
Test model without credit history to verify it considers financial factors
"""
import joblib
import pandas as pd

# Load model, scaler, columns, and label encoders
model = joblib.load('models/loan_model.pkl')
scaler = joblib.load('models/scaler.pkl')
columns = joblib.load('models/columns.pkl')
label_encoders = joblib.load('models/label_encoders.pkl')

print(f"Model features: {len(columns)}")
print(f"Feature names: {columns}")
print()

# Test Case 1: Good financial factors (should be approved)
print("=" * 50)
print("Test Case 1: Good Financial Factors (Should be Approved)")
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
    'Property_Area': ['Urban']
}

df = pd.DataFrame(good_data)
df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['Income_Loan_Ratio'] = df['TotalIncome'] / (df['LoanAmount'] + 1)
df['EMI'] = df['LoanAmount'] / df['Loan_Amount_Term']
df['RiskScore'] = df['LoanAmount'] / (df['TotalIncome'] + 1)

print(f"Engineered features - TotalIncome: {df['TotalIncome'].values[0]}, "
      f"Income_Loan_Ratio: {df['Income_Loan_Ratio'].values[0]:.4f}, "
      f"EMI: {df['EMI'].values[0]:.2f}, "
      f"RiskScore: {df['RiskScore'].values[0]:.4f}")

# Encode categorical variables
for col in df.select_dtypes(include='object').columns:
    if col in label_encoders:
        df[col] = label_encoders[col].transform(df[col])

# Reindex to match training columns
df = df.reindex(columns=columns, fill_value=0)

# Scale
array = scaler.transform(df)

# Predict
prediction_proba = model.predict_proba(array)[0]
approval_threshold = 0.30
if prediction_proba[1] >= approval_threshold:
    prediction = 1
else:
    prediction = 0

print(f"Approval probability: {prediction_proba[1]:.4f}")
print(f"Prediction: {'Approved' if prediction == 1 else 'Rejected'}")
print()

# Test Case 2: Bad financial factors (should be rejected)
print("=" * 50)
print("Test Case 2: Bad Financial Factors (Should be Rejected)")
print("=" * 50)
bad_data = {
    'Gender': ['Male'],
    'Married': ['No'],
    'Dependents': ['0'],
    'Education': ['Graduate'],
    'Self_Employed': ['No'],
    'ApplicantIncome': [10000],
    'CoapplicantIncome': [0],
    'LoanAmount': [900000],
    'Loan_Amount_Term': [12],
    'Property_Area': ['Urban']
}

df = pd.DataFrame(bad_data)
df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['Income_Loan_Ratio'] = df['TotalIncome'] / (df['LoanAmount'] + 1)
df['EMI'] = df['LoanAmount'] / df['Loan_Amount_Term']
df['RiskScore'] = df['LoanAmount'] / (df['TotalIncome'] + 1)

print(f"Engineered features - TotalIncome: {df['TotalIncome'].values[0]}, "
      f"Income_Loan_Ratio: {df['Income_Loan_Ratio'].values[0]:.4f}, "
      f"EMI: {df['EMI'].values[0]:.2f}, "
      f"RiskScore: {df['RiskScore'].values[0]:.4f}")

# Encode categorical variables
for col in df.select_dtypes(include='object').columns:
    if col in label_encoders:
        df[col] = label_encoders[col].transform(df[col])

# Reindex to match training columns
df = df.reindex(columns=columns, fill_value=0)

# Scale
array = scaler.transform(df)

# Predict
prediction_proba = model.predict_proba(array)[0]
approval_threshold = 0.30
if prediction_proba[1] >= approval_threshold:
    prediction = 1
else:
    prediction = 0

print(f"Approval probability: {prediction_proba[1]:.4f}")
print(f"Prediction: {'Approved' if prediction == 1 else 'Rejected'}")
