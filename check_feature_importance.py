"""
Check feature importance of the current model
"""
import joblib
import pandas as pd

# Load model and columns
model = joblib.load('models/loan_model.pkl')
columns = joblib.load('models/columns.pkl')

# Get feature importance
feature_importance = model.feature_importances_

# Create DataFrame for better visualization
importance_df = pd.DataFrame({
    'Feature': columns,
    'Importance': feature_importance
})

# Sort by importance
importance_df = importance_df.sort_values('Importance', ascending=False)

print("Feature Importance:")
print(importance_df.to_string(index=False))
print()

# Check if credit_history is dominating
credit_history_importance = importance_df[importance_df['Feature'] == 'Credit_History']['Importance'].values[0]
print(f"Credit_History importance: {credit_history_importance:.4f}")
print(f"Total importance: {importance_df['Importance'].sum():.4f}")
print(f"Credit_History percentage: {credit_history_importance/importance_df['Importance'].sum()*100:.2f}%")
