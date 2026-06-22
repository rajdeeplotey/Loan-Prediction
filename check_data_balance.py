"""
Check training data balance
"""
import pandas as pd

# Load the dataset
df = pd.read_csv("Loan_Train.csv")

# Check loan status distribution
print("Loan Status Distribution:")
print(df['Loan_Status'].value_counts())
print()

# Calculate percentages
print("Loan Status Percentage:")
print(df['Loan_Status'].value_counts(normalize=True) * 100)
print()

# Check if data is imbalanced
approved = len(df[df['Loan_Status'] == 'Y'])
rejected = len(df[df['Loan_Status'] == 'N'])
total = len(df)

print(f"Total applications: {total}")
print(f"Approved: {approved} ({approved/total*100:.2f}%)")
print(f"Rejected: {rejected} ({rejected/total*100:.2f}%)")
print()

if approved/total > 0.7 or rejected/total > 0.7:
    print("WARNING: Data is imbalanced!")
else:
    print("Data is reasonably balanced")
