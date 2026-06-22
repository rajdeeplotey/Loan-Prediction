"""
Loan Approval Prediction System - Flask Backend
A production-ready Flask application for predicting loan approval status
using a trained Random Forest Classifier model.
"""

import os
import logging
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, session
import joblib
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Configure SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///loan_applications.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy
db = SQLAlchemy(app)


# Database Model for Loan Applications
class LoanApplication(db.Model):
    """
    Model for storing loan application predictions.
    """
    __tablename__ = 'loan_applications'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=True)
    gender = db.Column(db.String(10), nullable=False)
    married = db.Column(db.String(10), nullable=False)
    dependents = db.Column(db.String(10), nullable=False)
    education = db.Column(db.String(20), nullable=False)
    self_employed = db.Column(db.String(10), nullable=False)
    applicant_income = db.Column(db.Float, nullable=False)
    coapplicant_income = db.Column(db.Float, nullable=False)
    loan_amount = db.Column(db.Float, nullable=False)
    loan_term = db.Column(db.Float, nullable=False)
    credit_history = db.Column(db.Integer, nullable=False)
    property_area = db.Column(db.String(20), nullable=False)
    prediction = db.Column(db.String(20), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    risk_level = db.Column(db.String(20), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<LoanApplication {self.id} - {self.prediction}>'


def login_required(f):
    """
    Decorator to require login for accessing routes.
    Redirects to login page if user is not authenticated.
    """
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this feature.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

# Global variables for model, columns, scaler, and label encoders
model = None
columns = None
scaler = None
label_encoders = None


def load_model():
    """
    Load the trained Random Forest model from disk.
    
    Returns:
        RandomForestClassifier: The loaded model object
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        Exception: For any other loading errors
    """
    global model
    try:
        # Use absolute path for cross-platform compatibility
        base_dir = Path(__file__).parent
        model_path = base_dir / 'models' / 'loan_model.pkl'
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        logger.info(f"Loading model from: {model_path}")
        model = joblib.load(model_path)
        logger.info("Model loaded successfully")
        logger.info(f"Model type: {type(model)}")
        logger.info(f"Model expected features: {model.n_features_in_ if hasattr(model, 'n_features_in_') else 'unknown'}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.error(f"Current working directory: {os.getcwd()}")
        logger.error(f"Script location: {Path(__file__).parent}")
        raise


def load_columns():
    """
    Load the training columns from disk.
    
    Returns:
        list: The list of column names
        
    Raises:
        FileNotFoundError: If columns file doesn't exist
        Exception: For any other loading errors
    """
    global columns
    try:
        # Use absolute path for cross-platform compatibility
        base_dir = Path(__file__).parent
        columns_path = base_dir / 'models' / 'columns.pkl'
        
        if not columns_path.exists():
            raise FileNotFoundError(f"Columns file not found at {columns_path}")
        
        logger.info(f"Loading columns from: {columns_path}")
        columns = joblib.load(columns_path)
        logger.info(f"Columns loaded successfully: {len(columns)} columns")
        logger.info(f"Column names: {columns}")
        return columns
    except Exception as e:
        logger.error(f"Error loading columns: {str(e)}")
        logger.error(f"Current working directory: {os.getcwd()}")
        logger.error(f"Script location: {Path(__file__).parent}")
        raise


def load_label_encoders():
    """
    Load the LabelEncoders from disk.
    
    Returns:
        dict: Dictionary of LabelEncoders for each categorical column
        
    Raises:
        FileNotFoundError: If label encoders file doesn't exist
        Exception: For any other loading errors
    """
    global label_encoders
    try:
        # Use absolute path for cross-platform compatibility
        base_dir = Path(__file__).parent
        le_path = base_dir / 'models' / 'label_encoders.pkl'
        
        if not le_path.exists():
            raise FileNotFoundError(f"Label encoders file not found at {le_path}")
        
        logger.info(f"Loading label encoders from: {le_path}")
        label_encoders = joblib.load(le_path)
        logger.info(f"Label encoders loaded successfully for columns: {list(label_encoders.keys())}")
        return label_encoders
    except Exception as e:
        logger.error(f"Error loading label encoders: {str(e)}")
        logger.error(f"Current working directory: {os.getcwd()}")
        logger.error(f"Script location: {Path(__file__).parent}")
        raise


def load_scaler():
    """
    Load the StandardScaler from disk.
    
    Returns:
        StandardScaler: The loaded scaler object
        
    Raises:
        FileNotFoundError: If scaler file doesn't exist
        Exception: For any other loading errors
    """
    global scaler
    try:
        # Use absolute path for cross-platform compatibility
        base_dir = Path(__file__).parent
        scaler_path = base_dir / 'models' / 'scaler.pkl'
        
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
        
        logger.info(f"Loading scaler from: {scaler_path}")
        scaler = joblib.load(scaler_path)
        logger.info("Scaler loaded successfully")
        return scaler
    except Exception as e:
        logger.error(f"Error loading scaler: {str(e)}")
        logger.error(f"Current working directory: {os.getcwd()}")
        logger.error(f"Script location: {Path(__file__).parent}")
        raise


# Create database tables
with app.app_context():
    db.create_all()
    logger.info("Database tables created successfully")


def validate_input(data):
    """
    Validate user input data for loan application.
    
    Args:
        data (dict): Dictionary containing form data
        
    Returns:
        tuple: (is_valid, error_message)
        
    Raises:
        ValueError: If validation fails
    """
    try:
        # Validate required fields exist
        required_fields = ['gender', 'married', 'dependents', 'education', 'self_employed', 
                         'applicant_income', 'loan_amount', 'loan_term_years', 'credit_history', 
                         'property_area']
        
        for field in required_fields:
            if field not in data or data[field] is None or data[field] == '':
                logger.error(f"Missing required field: {field}")
                return False, f"Missing required field: {field.replace('_', ' ').title()}"
        
        # Validate Applicant Income
        applicant_income_str = data.get('applicant_income', '0')
        if not applicant_income_str or applicant_income_str == '':
            return False, "Applicant Income is required"
        
        applicant_income = float(applicant_income_str)
        if applicant_income <= 0:
            return False, "Applicant Income must be greater than 0"
        
        # Validate Coapplicant Income (optional)
        coapplicant_income_str = data.get('coapplicant_income', '0')
        if coapplicant_income_str and coapplicant_income_str != '':
            coapplicant_income = float(coapplicant_income_str)
            if coapplicant_income < 0:
                return False, "Coapplicant Income cannot be negative"
        
        # Validate Loan Amount
        loan_amount_str = data.get('loan_amount', '0')
        if not loan_amount_str or loan_amount_str == '':
            return False, "Loan Amount is required"
        
        loan_amount = float(loan_amount_str)
        if loan_amount <= 0:
            return False, "Loan Amount must be greater than 0"
        
        # Validate Loan Term
        loan_term_str = data.get('loan_term_years', '0')
        if not loan_term_str or loan_term_str == '':
            return False, "Loan Term is required"
        
        loan_term = float(loan_term_str)
        if loan_term <= 0:
            return False, "Loan Term must be greater than 0"
        
        # Validate Credit History (original field)
        credit_history_str = data.get('credit_history', '0')
        if not credit_history_str or credit_history_str == '':
            return False, "Credit History is required"
        
        credit_history = int(credit_history_str)
        if credit_history not in [0, 1]:
            return False, "Credit History must be 0 or 1"
        
        # Validate dropdown values
        valid_genders = ['Male', 'Female']
        if data.get('gender') not in valid_genders:
            return False, "Invalid Gender value"
        
        valid_marital_status = ['Yes', 'No']
        if data.get('married') not in valid_marital_status:
            return False, "Invalid Marital Status value"
        
        valid_dependents = ['0', '1', '2', '3+']
        if data.get('dependents') not in valid_dependents:
            return False, "Invalid Dependents value"
        
        valid_education = ['Graduate', 'Not Graduate']
        if data.get('education') not in valid_education:
            return False, "Invalid Education value"
        
        valid_employment = ['Yes', 'No']
        if data.get('self_employed') not in valid_employment:
            return False, "Invalid Employment Status value"
        
        valid_property_areas = ['Urban', 'Semiurban', 'Rural']
        if data.get('property_area') not in valid_property_areas:
            return False, "Invalid Property Area value"
        
        return True, None
        
    except (ValueError, TypeError) as e:
        logger.error(f"Input validation error: {str(e)}")
        logger.error(f"Form data received: {data}")
        return False, f"Invalid input format: {str(e)}"


def prepare_input_data(form_data):
    """
    Prepare input data for prediction using the same feature engineering as training.
    
    Args:
        form_data (dict): Dictionary containing form data
        
    Returns:
        numpy.ndarray: Prepared array ready for prediction
    """
    try:
        # Convert loan term from years to months
        loan_term_years = int(form_data.get('loan_term_years', 30))
        loan_term_months = loan_term_years * 12
        
        # Get numeric values
        applicant_income = float(form_data.get('applicant_income', 0))
        coapplicant_income = float(form_data.get('coapplicant_income', 0))
        loan_amount = float(form_data.get('loan_amount', 0))
        
        # Create a dictionary with the exact column names expected by the model
        # These should match the training data column names
        input_dict = {
            'Gender': [form_data.get('gender', 'Male')],
            'Married': [form_data.get('married', 'No')],
            'Dependents': [form_data.get('dependents', '0')],
            'Education': [form_data.get('education', 'Graduate')],
            'Self_Employed': [form_data.get('self_employed', 'No')],
            'ApplicantIncome': [applicant_income],
            'CoapplicantIncome': [coapplicant_income],
            'LoanAmount': [loan_amount],
            'Loan_Amount_Term': [float(loan_term_months)],
            'Credit_History': [int(form_data.get('credit_history', 1))],
            'Property_Area': [form_data.get('property_area', 'Urban')]
        }
        
        logger.info(f"Input dict: {input_dict}")
        
        # Create DataFrame
        input_df = pd.DataFrame(input_dict)
        logger.info(f"DataFrame created: {input_df.shape}")
        
        # Apply the SAME feature engineering as training
        input_df['TotalIncome'] = input_df['ApplicantIncome'] + input_df['CoapplicantIncome']
        input_df['Income_Loan_Ratio'] = input_df['TotalIncome'] / (input_df['LoanAmount'] + 1)
        input_df['EMI'] = input_df['LoanAmount'] / input_df['Loan_Amount_Term']
        input_df['RiskScore'] = input_df['LoanAmount'] / (input_df['TotalIncome'] + 1)
        
        logger.info(f"After feature engineering: {input_df.shape}")
        logger.info(f"Engineered features - TotalIncome: {input_df['TotalIncome'].values[0]}, "
                   f"Income_Loan_Ratio: {input_df['Income_Loan_Ratio'].values[0]:.4f}, "
                   f"EMI: {input_df['EMI'].values[0]:.2f}, "
                   f"RiskScore: {input_df['RiskScore'].values[0]:.4f}")
        
        # Encode categorical variables using the SAME LabelEncoders from training
        if label_encoders is not None:
            for col in input_df.select_dtypes(include='object').columns:
                if col in label_encoders:
                    input_df[col] = label_encoders[col].transform(input_df[col])
                    logger.info(f"Encoded {col} using saved LabelEncoder")
                else:
                    logger.warning(f"No LabelEncoder found for {col}, using default encoding")
                    le = LabelEncoder()
                    input_df[col] = le.fit_transform(input_df[col])
        else:
            logger.warning("Label encoders not loaded, using new LabelEncoder (may cause incorrect predictions)")
            le = LabelEncoder()
            for col in input_df.select_dtypes(include='object').columns:
                input_df[col] = le.fit_transform(input_df[col])
        
        logger.info(f"After encoding: {input_df.shape}, columns: {input_df.columns.tolist()}")
        
        # Reindex to match training columns
        input_df = input_df.reindex(columns=columns, fill_value=0)
        logger.info(f"After reindexing: {input_df.shape}")
        
        # Apply scaling (same as training)
        if scaler is not None:
            input_array = scaler.transform(input_df)
            logger.info(f"After scaling: {input_array.shape}")
        else:
            input_array = input_df.values
            logger.warning("Scaler not loaded, using unscaled values")
        
        # Validate feature count matches model expectations
        if model is not None and hasattr(model, 'n_features_in_'):
            expected_features = model.n_features_in_
            actual_features = input_array.shape[1]
            if expected_features != actual_features:
                logger.error(f"FEATURE COUNT MISMATCH: Model expects {expected_features} features but input has {actual_features} features")
                logger.error(f"Expected features from model: {expected_features}")
                logger.error(f"Actual features from input: {actual_features}")
                logger.error(f"Columns file has: {len(columns)} features")
                logger.error(f"Input columns: {input_df.columns.tolist()}")
                raise ValueError(f"Feature count mismatch: Model expects {expected_features} features but input has {actual_features} features")
        
        logger.info(f"Input data prepared successfully. Shape: {input_array.shape}")
        return input_array
        
    except Exception as e:
        logger.error(f"Error preparing input data: {str(e)}")
        logger.error(f"Form data: {form_data}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error details: {str(e)}", exc_info=True)
        raise


def determine_risk_level(confidence):
    """
    Determine risk level based on confidence score.
    
    Args:
        confidence (float): Confidence score from model prediction (0-100)
        
    Returns:
        str: Risk level category
    """
    if confidence >= 80:
        return "Low Risk"
    elif confidence >= 60:
        return "Medium Risk"
    else:
        return "High Risk"


def get_recommendation(prediction):
    """
    Get recommendation message based on prediction result.
    
    Args:
        prediction (int): Model prediction (0 or 1)
        
    Returns:
        str: Recommendation message
    """
    if prediction == 1:
        return "Applicant appears financially eligible for loan approval."
    else:
        return "Applicant may improve credit history or income profile before reapplying."


def get_approval_factors(form_data):
    """
    Generate approval factors based on submitted values.
    
    Args:
        form_data (dict): Dictionary containing form data
        
    Returns:
        list: List of approval factors
    """
    approval_factors = []
    
    # Good Credit History
    if int(form_data.get('credit_history', 0)) == 1:
        approval_factors.append("Good Credit History")
    
    # Graduate Education
    if form_data.get('education') == 'Graduate':
        approval_factors.append("Graduate Education")
    
    # Strong Income
    applicant_income = float(form_data.get('applicant_income', 0))
    coapplicant_income = float(form_data.get('coapplicant_income', 0))
    total_income = applicant_income + coapplicant_income
    if total_income >= 50000:
        approval_factors.append("Strong Income")
    
    # Low Loan Amount
    loan_amount = float(form_data.get('loan_amount', 0))
    if loan_amount <= 500000:
        approval_factors.append("Low Loan Amount")
    
    # Urban/Semiurban Property
    if form_data.get('property_area') in ['Urban', 'Semiurban']:
        approval_factors.append(f"{form_data.get('property_area')} Property")
    
    # No Dependents
    if form_data.get('dependents') == '0':
        approval_factors.append("No Dependents")
    
    # Married
    if form_data.get('married') == 'Yes':
        approval_factors.append("Married Status")
    
    # Not Self Employed
    if form_data.get('self_employed') == 'No':
        approval_factors.append("Salaried Employment")
    
    return approval_factors


def get_risk_factors(form_data):
    """
    Generate risk factors based on submitted values.
    
    Args:
        form_data (dict): Dictionary containing form data
        
    Returns:
        list: List of risk factors
    """
    risk_factors = []
    
    # Poor Credit History
    if int(form_data.get('credit_history', 0)) == 0:
        risk_factors.append("Poor Credit History")
    
    # Low Income
    applicant_income = float(form_data.get('applicant_income', 0))
    coapplicant_income = float(form_data.get('coapplicant_income', 0))
    total_income = applicant_income + coapplicant_income
    if total_income < 20000:
        risk_factors.append("Low Income")
    
    # High Loan Amount
    loan_amount = float(form_data.get('loan_amount', 0))
    if loan_amount > 1000000:
        risk_factors.append("High Loan Amount")
    
    # Multiple Dependents
    if form_data.get('dependents') in ['2', '3+']:
        risk_factors.append("Multiple Dependents")
    
    # No Coapplicant Income
    if coapplicant_income == 0:
        risk_factors.append("No Coapplicant Income")
    
    # Rural Property
    if form_data.get('property_area') == 'Rural':
        risk_factors.append("Rural Property")
    
    # Self Employed
    if form_data.get('self_employed') == 'Yes':
        risk_factors.append("Self Employment")
    
    # Not Graduate
    if form_data.get('education') != 'Graduate':
        risk_factors.append("Non-Graduate Education")
    
    # Unmarried
    if form_data.get('married') == 'No':
        risk_factors.append("Unmarried Status")
    
    return risk_factors


@app.route('/')
def index():
    """
    Render the homepage.
    
    Returns:
        str: Rendered HTML template
    """
    logger.info("Homepage accessed")
    return render_template('home.html')


@app.route('/home')
def home():
    """
    Render the homepage.
    
    Returns:
        str: Rendered HTML template
    """
    logger.info("Home page accessed")
    return render_template('home.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    """
    Handle user registration.
    
    Returns:
        str: Rendered HTML template
    """
    if request.method == 'POST':
        # Handle registration logic
        session['user_id'] = 'new_user'
        session['user_name'] = request.form.get('full_name', 'User')
        flash('Registration successful! Please verify your email.', 'success')
        return redirect(url_for('verify_email'))
    logger.info("Registration page accessed")
    return render_template('register.html')


@app.route('/verify_email', methods=['GET', 'POST'])
def verify_email():
    """
    Handle email verification.
    
    Returns:
        str: Rendered HTML template
    """
    if request.method == 'POST':
        # Handle OTP verification
        flash('Email verified successfully!', 'success')
        return redirect(url_for('login'))
    logger.info("Email verification page accessed")
    return render_template('verify_email.html')


@app.route('/logout')
def logout():
    """
    Handle user logout.
    
    Returns:
        str: Redirect to home page
    """
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    """
    Handle user login.
    
    Returns:
        str: Rendered HTML template
    """
    if request.method == 'POST':
        # Handle login logic
        # For demo purposes, accept any login
        session['user_id'] = 'demo_user'
        session['user_name'] = request.form.get('email', 'User')
        flash('Login successful!', 'success')
        return redirect(url_for('dashboard'))
    logger.info("Login page accessed")
    return render_template('login.html')


@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    """
    Handle forgot password request.
    
    Returns:
        str: Rendered HTML template
    """
    if request.method == 'POST':
        # Handle forgot password logic
        flash('Password reset link sent to your email.', 'success')
        return redirect(url_for('reset_password'))
    logger.info("Forgot password page accessed")
    return render_template('forgot_password.html')


@app.route('/reset_password', methods=['GET', 'POST'])
def reset_password():
    """
    Handle password reset.
    
    Returns:
        str: Rendered HTML template
    """
    if request.method == 'POST':
        # Handle password reset logic
        flash('Password reset successfully!', 'success')
        return redirect(url_for('login'))
    logger.info("Reset password page accessed")
    return render_template('forgot_password.html')


@app.route('/dashboard')
@login_required
def dashboard():
    """
    Render the user dashboard.
    
    Returns:
        str: Rendered HTML template
    """
    logger.info("Dashboard accessed")
    return render_template('dashboard.html')


@app.route('/loan_prediction')
def loan_prediction():
    """
    Render the loan prediction form page.
    
    Returns:
        str: Rendered HTML template
    """
    logger.info("Loan prediction page accessed")
    return render_template('loan_prediction.html')


@app.route('/prediction_result')
@login_required
def prediction_result():
    """
    Render the prediction result page.
    
    Returns:
        str: Rendered HTML template
    """
    logger.info("Prediction result page accessed")
    return render_template('prediction_result.html')


@app.route('/analytics')
@login_required
def analytics():
    """
    Render the analytics dashboard.
    
    Returns:
        str: Rendered HTML template
    """
    logger.info("Analytics dashboard accessed")
    return render_template('analytics.html')




@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    """
    Handle user profile page.
    
    Returns:
        str: Rendered HTML template
    """
    if request.method == 'POST':
        # Handle profile update
        flash('Profile updated successfully!', 'success')
    logger.info("Profile page accessed")
    return render_template('profile.html')


@app.route('/change_password', methods=['POST'])
def change_password():
    """
    Handle password change.
    
    Returns:
        str: Redirect to profile page
    """
    # Handle password change logic
    flash('Password changed successfully!', 'success')
    return redirect(url_for('profile'))


@app.route('/contact', methods=['GET', 'POST'])
@login_required
def contact():
    """
    Handle contact page.
    
    Returns:
        str: Rendered HTML template
    """
    if request.method == 'POST':
        # Handle contact form submission
        flash('Message sent successfully!', 'success')
    logger.info("Contact page accessed")
    return render_template('contact.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle loan prediction request.
    
    Receives form data, validates it, performs preprocessing,
    makes prediction, and returns results.
    
    Returns:
        str: Rendered result template with prediction details
    """
    try:
        # Get form data
        form_data = request.form.to_dict()
        logger.info(f"Prediction request received: {form_data}")
        
        # Validate input
        is_valid, error_message = validate_input(form_data)
        if not is_valid:
            logger.warning(f"Validation failed: {error_message}")
            return render_template('loan_prediction.html', error=error_message)
        
        # Prepare input data using Pandas DataFrame and one-hot encoding
        input_df = prepare_input_data(form_data)
        
        # Make prediction using the model directly
        prediction_proba = model.predict_proba(input_df)[0]
        
        # Use the model's prediction with standard threshold (0.50)
        approval_threshold = 0.50
        if prediction_proba[1] >= approval_threshold:
            prediction = 1
        else:
            prediction = 0
        
        confidence = max(prediction_proba) * 100  # Convert to percentage
        
        # Determine risk level
        risk_level = determine_risk_level(confidence)
        
        # Get recommendation
        recommendation = get_recommendation(prediction)
        
        # Convert prediction to readable format
        loan_status = "Approved" if prediction == 1 else "Rejected"
        
        # Get current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Generate approval and risk factors
        approval_factors = get_approval_factors(form_data)
        risk_factors = get_risk_factors(form_data)
        
        # Save prediction to database
        try:
            loan_term_years = int(form_data.get('loan_term_years', 30))
            loan_term_months = loan_term_years * 12
            
            application = LoanApplication(
                gender=form_data.get('gender'),
                married=form_data.get('married'),
                dependents=form_data.get('dependents'),
                education=form_data.get('education'),
                self_employed=form_data.get('self_employed'),
                applicant_income=float(form_data.get('applicant_income', 0)),
                coapplicant_income=float(form_data.get('coapplicant_income', 0)),
                loan_amount=float(form_data.get('loan_amount', 0)),
                loan_term=float(loan_term_months),
                credit_history=int(form_data.get('credit_history', 1)),
                property_area=form_data.get('property_area'),
                prediction=loan_status,
                confidence=float(confidence),
                risk_level=risk_level
            )
            db.session.add(application)
            db.session.commit()
            logger.info(f"Prediction saved to database with ID: {application.id}")
        except Exception as e:
            logger.error(f"Error saving prediction to database: {str(e)}")
            # Continue with the prediction even if database save fails
        
        # Log prediction
        logger.info(
            f"Prediction made - Status: {loan_status}, "
            f"Confidence: {confidence:.2f}%, Risk: {risk_level}, "
            f"Timestamp: {timestamp}"
        )
        
        # Prepare result data
        result = {
            'loan_status': loan_status,
            'confidence': f"{confidence:.2f}%",
            'risk_level': risk_level,
            'recommendation': recommendation,
            'timestamp': timestamp,
            'form_data': form_data,
            'approval_factors': approval_factors,
            'risk_factors': risk_factors
        }
        
        return render_template('prediction_result.html', result=result)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        error_message = "An error occurred during prediction. Please try again."
        return render_template('loan_prediction.html', error=error_message)


@app.route('/health')
def health():
    """
    Health check endpoint for monitoring.
    
    Returns:
        dict: JSON response with health status
    """
    return jsonify({"status": "healthy"})


@app.route('/api/analytics')
def get_analytics():
    """
    API endpoint to get analytics data from the database.
    
    Returns:
        JSON: Analytics data including total predictions, approval rate, etc.
    """
    try:
        total_predictions = LoanApplication.query.count()
        approved = LoanApplication.query.filter_by(prediction='Approved').count()
        rejected = LoanApplication.query.filter_by(prediction='Rejected').count()
        
        approval_rate = (approved / total_predictions * 100) if total_predictions > 0 else 0
        rejection_rate = (rejected / total_predictions * 100) if total_predictions > 0 else 0
        
        # Average confidence
        avg_confidence = db.session.query(db.func.avg(LoanApplication.confidence)).scalar() or 0
        
        # Risk distribution
        low_risk = LoanApplication.query.filter_by(risk_level='Low Risk').count()
        medium_risk = LoanApplication.query.filter_by(risk_level='Medium Risk').count()
        high_risk = LoanApplication.query.filter_by(risk_level='High Risk').count()
        
        analytics_data = {
            'total_predictions': total_predictions,
            'approved': approved,
            'rejected': rejected,
            'approval_rate': round(approval_rate, 2),
            'rejection_rate': round(rejection_rate, 2),
            'average_confidence': round(avg_confidence, 2),
            'risk_distribution': {
                'low_risk': low_risk,
                'medium_risk': medium_risk,
                'high_risk': high_risk
            }
        }
        
        return jsonify(analytics_data)
    except Exception as e:
        logger.error(f"Error fetching analytics: {str(e)}")
        return jsonify({'error': 'Failed to fetch analytics'}), 500


@app.errorhandler(404)
def not_found(error):
    """
    Handle 404 errors.
    
    Args:
        error: Error object
        
    Returns:
        str: Error message
    """
    logger.warning(f"404 error: {str(error)}")
    return "Page not found", 404


@app.errorhandler(500)
def internal_error(error):
    """
    Handle 500 errors.
    
    Args:
        error: Error object
        
    Returns:
        str: Error message
    """
    logger.error(f"500 error: {str(error)}")
    return "Internal server error", 500


def initialize_app():
    """
    Initialize the application by loading models, columns, scaler, and label encoders.
    This function is called when the application starts.
    """
    try:
        logger.info("Starting application initialization...")
        load_model()
        load_columns()
        load_scaler()
        load_label_encoders()
        
        # Validate model and columns compatibility
        if model is not None and columns is not None:
            if hasattr(model, 'n_features_in_'):
                model_features = model.n_features_in_
                columns_features = len(columns)
                logger.info(f"Model expects {model_features} features, columns file has {columns_features} features")
                
                if model_features != columns_features:
                    logger.error(f"FEATURE COUNT MISMATCH: Model expects {model_features} features but columns file has {columns_features} features")
                    raise ValueError(f"Feature count mismatch: Model expects {model_features} features but columns file has {columns_features} features")
        
        logger.info("Application initialized successfully")
    except Exception as e:
        logger.error(f"Application initialization failed: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error details: {str(e)}", exc_info=True)
        raise


# Initialize application on startup
initialize_app()


if __name__ == '__main__':
    # Run the Flask application
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=os.environ.get('DEBUG', 'False').lower() == 'true'
    )
