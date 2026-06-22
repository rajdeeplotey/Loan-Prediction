"""
Test model with good financial factors but different credit history values
"""
import joblib
import pandas as pd

# Load model, scaler, columns, and label encoders
model = joblib.load('models/loan_model.pkl')
scaler = joblib.load('models/scaler.pkl')
columns = joblib.load('models/columns.pkl')
label_encoders = joblib.load('models/label_encoders.pkl')

# Test applicant with good income, reasonable loan, good term
test_data = {
    'Gender': ['Male'],
    'Married': ['Yes'],
    'Dependents': ['0'],
    'Education': ['Graduate'],
    'Self_Employed': ['No'],
    'ApplicantIncome': [50000],  # Good income
    'CoapplicantIncome': [20000],
    'LoanAmount': [300000],  # Reasonable loan
    'Loan_Amount_Term': [360],  # 30 year term
    'Credit_History': [1],  # Good credit history
    'Property_Area': ['Urban']
}

df = pd.DataFrame(test_data)
df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['Income_Loan_Ratio'] = df['TotalIncome'] / (df['LoanAmount'] + 1)
df['EMI'] = df['LoanAmount'] / df['Loan_Amount_Term']
df['RiskScore'] = df['LoanAmount'] / (df['TotalIncome'] + 1)

print("Test Applicant: Good Income (70,000), Reasonable Loan (300,000), 30 Year Term")
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

# Test with Credit_History = 1
print("=" * 50)
print("Test 1: Credit_History = 1 (Good)")
print("=" * 50)
array = scaler.transform(df)
prediction_proba = model.predict_proba(array)[0]
approval_threshold = 0.30
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
approval_threshold = 0.30
if prediction_proba[1] >= approval_threshold:
    prediction = 1
else:
    prediction = 0
print(f"Approval probability: {prediction_proba[1]:.4f}")
print(f"Prediction: {'Approved' if prediction == 1 else 'Rejected'}")
