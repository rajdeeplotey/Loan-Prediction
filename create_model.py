"""
Script to train and save the Random Forest model for Loan Approval Prediction
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
df = pd.read_csv("Loan_Train.csv")

# Handle missing values
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median())
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])

# Feature Engineering
df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['Income_Loan_Ratio'] = df['TotalIncome'] / (df['LoanAmount'] + 1)
df['EMI'] = df['LoanAmount'] / df['Loan_Amount_Term']
df['RiskScore'] = df['LoanAmount'] / (df['TotalIncome'] + 1)

# Encode categorical variables
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Prepare features and target
X = df.drop(['Loan_ID', 'Loan_Status'], axis=1)
y = df['Loan_Status']

# Save column names before scaling
column_names = X.columns.tolist()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Random Forest model with simpler parameters to reduce feature dominance
rf = RandomForestClassifier(
    n_estimators=100,  # Reduce number of trees
    max_depth=8,  # Reduce depth
    min_samples_split=10,  # Increase min samples split
    min_samples_leaf=4,  # Increase min samples leaf
    random_state=42,
    class_weight='balanced'  # Add class weights to handle imbalance
)
rf.fit(X_train, y_train)

# Save the model, scaler, columns, and label encoders with correct names
joblib.dump(rf, "models/loan_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(column_names, "models/columns.pkl")
joblib.dump(label_encoders, "models/label_encoders.pkl")

print("Model, scaler, columns, and label encoders saved successfully!")
print(f"Train Accuracy: {rf.score(X_train, y_train):.4f}")
print(f"Test Accuracy: {rf.score(X_test, y_test):.4f}")
print(f"Number of features: {len(column_names)}")
print(f"Feature names: {column_names}")
print(f"Label encoders saved for columns: {list(label_encoders.keys())}")
