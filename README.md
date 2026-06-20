# Loan Approval Prediction System

A production-ready Flask backend for predicting loan approval status using a trained Random Forest Classifier model. This system provides an AI-powered loan eligibility assessment with confidence scores, risk categorization, and personalized recommendations.

## 🚀 Features

- **Machine Learning Powered**: Uses a Random Forest Classifier with ~87% test accuracy
- **Real-time Predictions**: Instant loan approval/rejection predictions
- **Confidence Scoring**: Provides model confidence percentage for each prediction
- **Risk Categorization**: Classifies applications as Low, Medium, or High Risk
- **Smart Recommendations**: Offers actionable advice based on prediction results
- **Input Validation**: Comprehensive validation of all user inputs
- **Responsive Design**: Modern, mobile-friendly user interface
- **Production Ready**: Configured for deployment on Render, Railway, and Hugging Face Spaces

## 📋 Project Structure

```
loan_approval_project/
├── app.py                      # Main Flask application
├── models/
│   ├── model.pkl              # Trained Random Forest model
│   └── scaler.pkl             # StandardScaler for feature scaling
├── templates/
│   ├── index.html             # Loan application form
│   └── result.html            # Prediction results page
├── static/
│   ├── css/
│   │   └── style.css          # Application styles
│   └── js/
│       └── script.js          # Client-side JavaScript
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── create_model.py           # Script to train and save model
└── Loan_Train.csv            # Training dataset
```

## 🛠️ Tech Stack

- **Backend**: Python 3.11+, Flask 3.0.0
- **Machine Learning**: Scikit-Learn 1.3.2, Random Forest Classifier
- **Data Processing**: Pandas 2.1.4, NumPy 1.26.2
- **Model Persistence**: Joblib 1.3.2
- **Production Server**: Gunicorn 21.2.0

## 📦 Installation

### Prerequisites

- Python 3.11 or higher
- pip (Python package manager)

### Setup Instructions

1. **Clone or navigate to the project directory**
   ```bash
   cd loan_approval_project
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify model files exist**
   The `models/` directory should contain:
   - `model.pkl` (trained Random Forest model)
   - `scaler.pkl` (StandardScaler)

   If these files don't exist, run:
   ```bash
   python create_model.py
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Access the application**
   Open your browser and navigate to: `http://localhost:5000`

## 🎯 Model Performance

- **Algorithm**: Random Forest Classifier
- **Training Accuracy**: ~89.6%
- **Test Accuracy**: ~87%
- **Features**: 15 engineered features including income ratios, EMI calculations, and risk scores

## 🔧 Configuration

### Environment Variables

Set the following environment variables for production:

- `SECRET_KEY`: Flask secret key for session management
- `PORT`: Port number (default: 5000)
- `DEBUG`: Debug mode (default: False)

Example:
```bash
export SECRET_KEY="your-secret-key-here"
export PORT=5000
export DEBUG=False
```

## 📡 API Endpoints

### GET `/`
Renders the loan application form page.

### POST `/predict`
Accepts loan application data and returns prediction results.

**Request Parameters:**
- `gender`: Male/Female
- `married`: Yes/No
- `dependents`: 0, 1, 2, 3+
- `education`: Graduate/Not Graduate
- `self_employed`: Yes/No
- `applicant_income`: Annual income (must be > 0)
- `coapplicant_income`: Coapplicant income (must be >= 0)
- `loan_amount`: Loan amount (must be > 0)
- `loan_term`: Loan term in months (must be > 0)
- `credit_history`: 0 or 1
- `property_area`: Urban/Semiurban/Rural

**Response:**
- `loan_status`: Approved/Rejected
- `confidence`: Confidence score (0-100%)
- `risk_level`: Low Risk/Medium Risk/High Risk
- `recommendation`: Personalized recommendation message

### GET `/health`
Health check endpoint for monitoring.

**Response:**
```json
{
  "status": "healthy"
}
```

## 🧪 Testing the Application

### Manual Testing

1. Fill out the loan application form with valid data
2. Click "Check Loan Eligibility"
3. View prediction results with confidence score and risk level

### Sample Test Data

**Test Case 1 (Likely Approved):**
- Gender: Male
- Married: Yes
- Dependents: 1
- Education: Graduate
- Self Employed: No
- Applicant Income: 5000
- Coapplicant Income: 2000
- Loan Amount: 150
- Loan Term: 360
- Credit History: 1
- Property Area: Urban

**Test Case 2 (Likely Rejected):**
- Gender: Male
- Married: No
- Dependents: 0
- Education: Not Graduate
- Self Employed: Yes
- Applicant Income: 1000
- Coapplicant Income: 0
- Loan Amount: 200
- Loan Term: 360
- Credit History: 0
- Property Area: Rural

## 🚀 Deployment

### Deploy to Render

1. Create a `render.yaml` file:
```yaml
services:
  - type: web
    name: loan-approval-api
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn app:app
    envVars:
      - key: SECRET_KEY
        generateValue: true
      - key: PORT
        value: 5000
```

2. Push your code to GitHub
3. Connect your repository to Render
4. Deploy

### Deploy to Railway

1. Install Railway CLI:
```bash
npm install -g @railway/cli
```

2. Login and deploy:
```bash
railway login
railway init
railway up
```

3. Set environment variables in Railway dashboard

### Deploy to Hugging Face Spaces

1. Create a new Space on Hugging Face
2. Choose "Flask" as the SDK
3. Upload your project files
4. Add `requirements.txt`
5. Set `SECRET_KEY` in repository settings
6. Deploy

## 🔒 Security Considerations

- Input validation on both client and server side
- No hardcoded secrets in code
- Environment-based configuration
- Secure form handling
- Error handling without exposing sensitive information

## 📊 Feature Engineering

The model uses the following engineered features:

1. **TotalIncome**: Applicant Income + Coapplicant Income
2. **Income_Loan_Ratio**: TotalIncome / (LoanAmount + 1)
3. **EMI**: LoanAmount / Loan_Amount_Term
4. **RiskScore**: LoanAmount / (TotalIncome + 1)

These features help the model better assess loan eligibility by considering income-to-debt ratios and repayment capacity.

## 🐛 Troubleshooting

### Common Issues

**Issue**: Model file not found
- **Solution**: Run `python create_model.py` to generate model files

**Issue**: Port already in use
- **Solution**: Change the PORT environment variable or stop the conflicting process

**Issue**: Import errors
- **Solution**: Ensure all dependencies are installed: `pip install -r requirements.txt`

**Issue**: Prediction errors
- **Solution**: Check the application logs for detailed error messages

## 📝 License

This project is provided as-is for educational and commercial use.

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Support

For issues or questions, please open an issue in the project repository.

---

**Built with ❤️ using Flask and Machine Learning**
