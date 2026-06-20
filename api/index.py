"""
Vercel Serverless Function for Loan Prediction API
"""
import sys
import os
from pathlib import Path

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import app

# Vercel serverless function handler
def handler(request):
    """
    Vercel serverless function handler.
    
    Args:
        request: Vercel request object
        
    Returns:
        Response from Flask application
    """
    return app(request.environ, lambda status, headers: None)
