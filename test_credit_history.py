"""
Test model with same applicant but different credit history values
"""
import joblib
import pandas as pd

# Load model, scaler, columns, and label encoders
model = joblib.load('models/loan_model.pkl')
scaler = joblib.load('models/scaler.pkl')
columns = joblib.load('models/columns.pkl')
label_encoders = joblib.load('models/label_encoders.pkl')

# Test applicant with low income, high loan, short term
test_data = {
    'Gender': ['Male'],
    'Married': ['No'],
    'Dependents': ['0'],
    'Education': ['Graduate'],
    'Self_Employed': ['No'],
    'ApplicantIncome': [10000],  # Low income
    'CoapplicantIncome': [0],
    'LoanAmount': [900000],  # High loan
    'Loan_Amount_Term': [12],  # 1 year term
    'Credit_History': [1],  # Good credit history
    'Property_Area': ['Urban']
}

df = pd.DataFrame(test_data)
df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['Income_Loan_Ratio'] = df['TotalIncome'] / (df['LoanAmount'] + 1)
df['EMI'] = df['LoanAmount'] / df['Loan_Amount_Term']
df['RiskScore'] = df['LoanAmount'] / (df['TotalIncome'] + 1)

print("Test Applicant: Low Income (10,000), High Loan (900,000), 1 Year Term")
print(f"Engineered features - TotalIncome: {df['TotalIncome'].values[0]}, "
      f"Income_Loan_Ratio: {df['Income_Loan_Ratio'].values[0]:.4f}, "
      f"EMI: {df['EMI'].values[0]:.2f}, "
      f"RiskScore: {df['RiskScore'].values[0]:.4f}")
print()

# Encode categorical variables
for col in df.select_dtypes(include='object').columns:
    if col in label_encoders:
        df[col] = label_encoders[col].transform(df[col])

# Reindex to match training columns
df = df.reindex(columns=columns, fill_value=0)

# Scale
array = scaler.transform(df)

# Test with Credit_History = 1
print("=" * 50)
print("Test 1: Credit_History = 1 (Good)")
print("=" * 50)
prediction_proba = model.predict_proba(array)[0]
approval_threshold = 0.35
if prediction_proba[1] >= approval_threshold:
    prediction = 1
else:
    prediction = 0
print(f"Approval probability: {prediction_proba[1]:.4f}")
print(f"Prediction: {'Approved' if prediction == 1 else 'Rejected'}")
print()

# Test with Credit_History = 0
print("=" * 50)
print("Test 2: Credit_History = 0 (Bad)")
print("=" * 50)
df['Credit_History'] = 0
array = scaler.transform(df)
prediction_proba = model.predict_proba(array)[0]
approval_threshold = 0.35
if prediction_proba[1] >= approval_threshold:
    prediction = 1
else:
    prediction = 0
print(f"Approval probability: {prediction_proba[1]:.4f}")
print(f"Prediction: {'Approved' if prediction == 1 else 'Rejected'}")
