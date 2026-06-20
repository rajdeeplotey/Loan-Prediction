// Loan Approval Prediction System - JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Form validation and submission handling
    const loanForm = document.getElementById('loanForm');
    
    if (loanForm) {
        loanForm.addEventListener('submit', function(e) {
            // Additional client-side validation
            if (!validateForm()) {
                e.preventDefault();
                return false;
            }
            
            // Show loading state
            const submitBtn = loanForm.querySelector('button[type="submit"]');
            const originalText = submitBtn.textContent;
            submitBtn.textContent = 'Processing...';
            submitBtn.disabled = true;
            submitBtn.style.opacity = '0.7';
            
            // Form will submit normally
        });
    }
    
    // Add input formatting
    formatNumericInputs();
    
    // Add real-time validation feedback
    addValidationListeners();
});

/**
 * Validate the loan application form
 * @returns {boolean} True if form is valid, false otherwise
 */
function validateForm() {
    const form = document.getElementById('loanForm');
    const applicantIncome = parseFloat(form.querySelector('#applicant_income').value);
    const coapplicantIncome = parseFloat(form.querySelector('#coapplicant_income').value) || 0;
    const loanAmount = parseFloat(form.querySelector('#loan_amount').value);
    const loanTerm = parseFloat(form.querySelector('#loan_term').value);
    
    // Validate applicant income
    if (isNaN(applicantIncome) || applicantIncome <= 0) {
        alert('Applicant Income must be greater than 0');
        return false;
    }
    
    // Validate coapplicant income
    if (isNaN(coapplicantIncome) || coapplicantIncome < 0) {
        alert('Coapplicant Income cannot be negative');
        return false;
    }
    
    // Validate loan amount
    if (isNaN(loanAmount) || loanAmount <= 0) {
        alert('Loan Amount must be greater than 0');
        return false;
    }
    
    // Validate loan term
    if (isNaN(loanTerm) || loanTerm <= 0) {
        alert('Loan Term must be greater than 0');
        return false;
    }
    
    // Check if all required fields are filled
    const requiredFields = form.querySelectorAll('[required]');
    for (let field of requiredFields) {
        if (!field.value.trim()) {
            alert('Please fill in all required fields');
            field.focus();
            return false;
        }
    }
    
    return true;
}

/**
 * Format numeric inputs for better user experience
 */
function formatNumericInputs() {
    const numericInputs = document.querySelectorAll('input[type="number"]');
    
    numericInputs.forEach(input => {
        input.addEventListener('blur', function() {
            const value = parseFloat(this.value);
            if (!isNaN(value)) {
                // Format to 2 decimal places for monetary values
                if (this.id.includes('income') || this.id.includes('amount')) {
                    this.value = value.toFixed(2);
                }
            }
        });
    });
}

/**
 * Add real-time validation listeners to form fields
 */
function addValidationListeners() {
    const form = document.getElementById('loanForm');
    if (!form) return;
    
    const inputs = form.querySelectorAll('input, select');
    
    inputs.forEach(input => {
        input.addEventListener('change', function() {
            validateField(this);
        });
        
        input.addEventListener('blur', function() {
            validateField(this);
        });
    });
}

/**
 * Validate individual form field
 * @param {HTMLElement} field - The form field to validate
 */
function validateField(field) {
    // Remove existing error styling
    field.style.borderColor = '#e0e0e0';
    
    const value = field.value.trim();
    
    // Skip validation if field is not required and empty
    if (!field.hasAttribute('required') && !value) {
        return;
    }
    
    // Required field validation
    if (field.hasAttribute('required') && !value) {
        field.style.borderColor = '#dc3545';
        return;
    }
    
    // Numeric field validation
    if (field.type === 'number') {
        const numValue = parseFloat(value);
        
        if (isNaN(numValue)) {
            field.style.borderColor = '#dc3545';
            return;
        }
        
        // Specific validations
        if (field.id === 'applicant_income' && numValue <= 0) {
            field.style.borderColor = '#dc3545';
            return;
        }
        
        if (field.id === 'coapplicant_income' && numValue < 0) {
            field.style.borderColor = '#dc3545';
            return;
        }
        
        if (field.id === 'loan_amount' && numValue <= 0) {
            field.style.borderColor = '#dc3545';
            return;
        }
        
        if (field.id === 'loan_term' && numValue <= 0) {
            field.style.borderColor = '#dc3545';
            return;
        }
    }
    
    // Valid field
    field.style.borderColor = '#28a745';
}

/**
 * Calculate and display estimated EMI (optional feature)
 */
function calculateEMI() {
    const loanAmount = parseFloat(document.getElementById('loan_amount').value);
    const loanTerm = parseFloat(document.getElementById('loan_term').value);
    const interestRate = 10; // Assuming 10% annual interest rate
    
    if (loanAmount && loanTerm) {
        const monthlyRate = interestRate / 12 / 100;
        const emi = (loanAmount * monthlyRate * Math.pow(1 + monthlyRate, loanTerm)) / 
                    (Math.pow(1 + monthlyRate, loanTerm) - 1);
        
        return emi.toFixed(2);
    }
    
    return 0;
}

/**
 * Add tooltip support for form fields
 */
function addTooltips() {
    const tooltipElements = document.querySelectorAll('[data-tooltip]');
    
    tooltipElements.forEach(element => {
        element.addEventListener('mouseenter', function() {
            const tooltip = document.createElement('div');
            tooltip.className = 'tooltip';
            tooltip.textContent = this.getAttribute('data-tooltip');
            tooltip.style.position = 'absolute';
            tooltip.style.background = '#333';
            tooltip.style.color = 'white';
            tooltip.style.padding = '5px 10px';
            tooltip.style.borderRadius = '5px';
            tooltip.style.fontSize = '0.85rem';
            tooltip.style.zIndex = '1000';
            tooltip.style.whiteSpace = 'nowrap';
            
            document.body.appendChild(tooltip);
            
            const rect = this.getBoundingClientRect();
            tooltip.style.top = (rect.bottom + 5) + 'px';
            tooltip.style.left = rect.left + 'px';
            
            this.tooltipElement = tooltip;
        });
        
        element.addEventListener('mouseleave', function() {
            if (this.tooltipElement) {
                this.tooltipElement.remove();
                this.tooltipElement = null;
            }
        });
    });
}

/**
 * Auto-save form data to localStorage (optional feature)
 */
function saveFormData() {
    const form = document.getElementById('loanForm');
    if (!form) return;
    
    const formData = {};
    form.querySelectorAll('input, select').forEach(field => {
        if (field.type !== 'submit' && field.type !== 'reset') {
            formData[field.name] = field.value;
        }
    });
    
    localStorage.setItem('loanFormData', JSON.stringify(formData));
}

/**
 * Load saved form data from localStorage (optional feature)
 */
function loadFormData() {
    const savedData = localStorage.getItem('loanFormData');
    if (!savedData) return;
    
    const formData = JSON.parse(savedData);
    const form = document.getElementById('loanForm');
    
    Object.keys(formData).forEach(fieldName => {
        const field = form.querySelector(`[name="${fieldName}"]`);
        if (field) {
            field.value = formData[fieldName];
        }
    });
}

/**
 * Clear saved form data
 */
function clearFormData() {
    localStorage.removeItem('loanFormData');
}

// Initialize tooltips on page load
addTooltips();
