/* ============================================
   LoanAI - Premium Fintech Platform JavaScript
   Interactive Features & Animations
   ============================================ */

// ===== Initialize Application =====
document.addEventListener('DOMContentLoaded', function() {
    // Initialize AOS animations
    AOS.init({
        duration: 800,
        easing: 'ease-in-out',
        once: true,
        offset: 100
    });
    
    // Initialize all features
    initPageLoader();
    initScrollProgress();
    initBackToTop();
    initThemeToggle();
    initNavbarScroll();
    initPasswordStrength();
    initPasswordToggle();
    initFormValidation();
    initProgressIndicator();
    initFAQAccordion();
    initOTPCountdown();
    initFileUpload();
    initChartAnimations();
    initSmoothScroll();
    initNumberCounter();
});

// ===== Page Loader =====
function initPageLoader() {
    const loader = document.getElementById('pageLoader');
    if (loader) {
        window.addEventListener('load', function() {
            setTimeout(function() {
                loader.classList.add('hidden');
            }, 500);
        });
    }
}

// ===== Scroll Progress Bar =====
function initScrollProgress() {
    const progressBar = document.getElementById('scrollProgress');
    if (progressBar) {
        window.addEventListener('scroll', function() {
            const scrollTop = document.documentElement.scrollTop || document.body.scrollTop;
            const scrollHeight = document.documentElement.scrollHeight - document.documentElement.clientHeight;
            const progress = (scrollTop / scrollHeight) * 100;
            progressBar.style.width = progress + '%';
        });
    }
}

// ===== Back to Top Button =====
function initBackToTop() {
    const backToTop = document.getElementById('backToTop');
    if (backToTop) {
        window.addEventListener('scroll', function() {
            if (window.scrollY > 300) {
                backToTop.classList.add('visible');
            } else {
                backToTop.classList.remove('visible');
            }
        });
        
        backToTop.addEventListener('click', function() {
            window.scrollTo({
                top: 0,
                behavior: 'smooth'
            });
        });
    }
}

// ===== Dark/Light Theme Toggle =====
function initThemeToggle() {
    const themeToggle = document.getElementById('themeToggle');
    const themeIcon = document.getElementById('themeIcon');
    const html = document.documentElement;
    
    // Check for saved theme preference
    const savedTheme = localStorage.getItem('theme') || 'light';
    html.setAttribute('data-bs-theme', savedTheme);
    updateThemeIcon(savedTheme);
    
    if (themeToggle) {
        themeToggle.addEventListener('click', function() {
            const currentTheme = html.getAttribute('data-bs-theme');
            const newTheme = currentTheme === 'light' ? 'dark' : 'light';
            
            html.setAttribute('data-bs-theme', newTheme);
            localStorage.setItem('theme', newTheme);
            updateThemeIcon(newTheme);
        });
    }
    
    function updateThemeIcon(theme) {
        if (themeIcon) {
            themeIcon.className = theme === 'light' ? 'fas fa-moon' : 'fas fa-sun';
        }
    }
}

// ===== Navbar Scroll Effect =====
function initNavbarScroll() {
    const navbar = document.getElementById('mainNavbar');
    if (navbar) {
        window.addEventListener('scroll', function() {
            if (window.scrollY > 50) {
                navbar.classList.add('scrolled');
            } else {
                navbar.classList.remove('scrolled');
            }
        });
    }
}

// ===== Password Strength Indicator =====
function initPasswordStrength() {
    const passwordInputs = document.querySelectorAll('input[type="password"][data-strength="true"]');
    
    passwordInputs.forEach(function(input) {
        const strengthBar = input.parentElement.querySelector('.password-strength-bar');
        if (!strengthBar) return;
        
        input.addEventListener('input', function() {
            const password = this.value;
            const strength = calculatePasswordStrength(password);
            
            strengthBar.className = 'password-strength-bar';
            if (strength >= 80) {
                strengthBar.classList.add('strong');
            } else if (strength >= 50) {
                strengthBar.classList.add('medium');
            } else if (strength > 0) {
                strengthBar.classList.add('weak');
            }
        });
    });
}

function calculatePasswordStrength(password) {
    let strength = 0;
    
    if (password.length >= 8) strength += 20;
    if (password.length >= 12) strength += 20;
    if (/[a-z]/.test(password)) strength += 10;
    if (/[A-Z]/.test(password)) strength += 10;
    if (/[0-9]/.test(password)) strength += 10;
    if (/[^a-zA-Z0-9]/.test(password)) strength += 20;
    if (password.length >= 16) strength += 10;
    
    return Math.min(strength, 100);
}

// ===== Password Visibility Toggle =====
function initPasswordToggle() {
    const toggleButtons = document.querySelectorAll('.password-toggle');
    
    toggleButtons.forEach(function(button) {
        button.addEventListener('click', function() {
            const input = this.parentElement.querySelector('input[type="password"], input[type="text"]');
            const icon = this.querySelector('i');
            
            if (input.type === 'password') {
                input.type = 'text';
                icon.className = 'fas fa-eye-slash';
            } else {
                input.type = 'password';
                icon.className = 'fas fa-eye';
            }
        });
    });
}

// ===== Form Validation =====
function initFormValidation() {
    const forms = document.querySelectorAll('form[data-validate="true"]');
    
    forms.forEach(function(form) {
        form.addEventListener('submit', function(e) {
            let isValid = true;
            const inputs = form.querySelectorAll('input[required], select[required], textarea[required]');
            
            inputs.forEach(function(input) {
                if (!input.value.trim()) {
                    isValid = false;
                    showInputError(input, 'This field is required');
                } else {
                    clearInputError(input);
                }
                
                // Email validation
                if (input.type === 'email' && input.value) {
                    if (!isValidEmail(input.value)) {
                        isValid = false;
                        showInputError(input, 'Please enter a valid email address');
                    }
                }
                
                // Password confirmation
                if (input.type === 'password' && input.name === 'confirm_password') {
                    const password = form.querySelector('input[name="password"]');
                    if (password && input.value !== password.value) {
                        isValid = false;
                        showInputError(input, 'Passwords do not match');
                    }
                }
            });
            
            if (!isValid) {
                e.preventDefault();
            }
        });
    });
}

function showInputError(input, message) {
    const errorDiv = input.parentElement.querySelector('.error-message');
    if (errorDiv) {
        errorDiv.textContent = message;
        errorDiv.style.display = 'block';
    }
    input.classList.add('is-invalid');
}

function clearInputError(input) {
    const errorDiv = input.parentElement.querySelector('.error-message');
    if (errorDiv) {
        errorDiv.style.display = 'none';
    }
    input.classList.remove('is-invalid');
}

function isValidEmail(email) {
    const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return re.test(email);
}

// ===== Progress Indicator =====
function initProgressIndicator() {
    const progressIndicators = document.querySelectorAll('.progress-indicator');
    
    progressIndicators.forEach(function(indicator) {
        const steps = indicator.querySelectorAll('.progress-step');
        const form = indicator.closest('form');
        
        if (form) {
            updateProgress(0, steps);
            
            form.addEventListener('submit', function(e) {
                const currentStep = getCurrentStep(steps);
                if (currentStep < steps.length - 1) {
                    e.preventDefault();
                    updateProgress(currentStep + 1, steps);
                }
            });
        }
    });
}

function getCurrentStep(steps) {
    for (let i = 0; i < steps.length; i++) {
        if (steps[i].classList.contains('active')) {
            return i;
        }
    }
    return 0;
}

function updateProgress(activeIndex, steps) {
    steps.forEach(function(step, index) {
        step.classList.remove('active', 'completed');
        
        if (index < activeIndex) {
            step.classList.add('completed');
        } else if (index === activeIndex) {
            step.classList.add('active');
        }
    });
}

// ===== FAQ Accordion =====
function initFAQAccordion() {
    const faqQuestions = document.querySelectorAll('.faq-question');
    
    faqQuestions.forEach(function(question) {
        question.addEventListener('click', function() {
            const answer = this.nextElementSibling;
            const isActive = this.classList.contains('active');
            
            // Close all other FAQs
            faqQuestions.forEach(function(q) {
                q.classList.remove('active');
                if (q.nextElementSibling) {
                    q.nextElementSibling.classList.remove('show');
                }
            });
            
            // Toggle current FAQ
            if (!isActive) {
                this.classList.add('active');
                if (answer) {
                    answer.classList.add('show');
                }
            }
        });
    });
}

// ===== OTP Countdown Timer =====
function initOTPCountdown() {
    const countdownElements = document.querySelectorAll('[data-countdown]');
    
    countdownElements.forEach(function(element) {
        let timeLeft = parseInt(element.dataset.countdown);
        
        const timer = setInterval(function() {
            timeLeft--;
            element.textContent = formatTime(timeLeft);
            
            if (timeLeft <= 0) {
                clearInterval(timer);
                element.textContent = 'Expired';
                element.classList.add('text-danger');
            }
        }, 1000);
    });
}

function formatTime(seconds) {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
}

// ===== File Upload with Drag & Drop =====
function initFileUpload() {
    const uploadAreas = document.querySelectorAll('.upload-area');
    
    uploadAreas.forEach(function(area) {
        const input = area.querySelector('input[type="file"]');
        
        // Drag and drop events
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            area.addEventListener(eventName, function(e) {
                e.preventDefault();
                e.stopPropagation();
            });
        });
        
        ['dragenter', 'dragover'].forEach(eventName => {
            area.addEventListener(eventName, function() {
                area.classList.add('dragover');
            });
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            area.addEventListener(eventName, function() {
                area.classList.remove('dragover');
            });
        });
        
        area.addEventListener('drop', function(e) {
            const files = e.dataTransfer.files;
            if (input) {
                input.files = files;
                handleFileSelect(files[0], area);
            }
        });
        
        // Click to upload
        area.addEventListener('click', function() {
            if (input) {
                input.click();
            }
        });
        
        // File input change
        if (input) {
            input.addEventListener('change', function() {
                if (this.files[0]) {
                    handleFileSelect(this.files[0], area);
                }
            });
        }
    });
}

function handleFileSelect(file, area) {
    const fileName = area.querySelector('.file-name');
    const fileSize = area.querySelector('.file-size');
    
    if (fileName) {
        fileName.textContent = file.name;
    }
    
    if (fileSize) {
        fileSize.textContent = formatFileSize(file.size);
    }
    
    area.classList.add('file-selected');
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

// ===== Chart Animations =====
function initChartAnimations() {
    // Initialize charts when they come into view
    const chartElements = document.querySelectorAll('canvas');
    
    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(function(entry) {
            if (entry.isIntersecting) {
                const canvas = entry.target;
                if (canvas.dataset.chartType) {
                    createChart(canvas);
                }
                observer.unobserve(canvas);
            }
        });
    }, { threshold: 0.5 });
    
    chartElements.forEach(function(canvas) {
        observer.observe(canvas);
    });
}

function createChart(canvas) {
    const ctx = canvas.getContext('2d');
    const chartType = canvas.dataset.chartType;
    const chartData = JSON.parse(canvas.dataset.chartData || '{}');
    
    const config = {
        type: chartType,
        data: chartData,
        options: {
            responsive: true,
            maintainAspectRatio: true,
            animation: {
                duration: 1000,
                easing: 'easeInOutQuart'
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'bottom'
                }
            }
        }
    };
    
    new Chart(ctx, config);
}

// ===== Smooth Scroll =====
function initSmoothScroll() {
    const links = document.querySelectorAll('a[href^="#"]');
    
    links.forEach(function(link) {
        link.addEventListener('click', function(e) {
            const href = this.getAttribute('href');
            if (href === '#') return;
            
            const target = document.querySelector(href);
            if (target) {
                e.preventDefault();
                const headerOffset = 80;
                const elementPosition = target.getBoundingClientRect().top;
                const offsetPosition = elementPosition + window.pageYOffset - headerOffset;
                
                window.scrollTo({
                    top: offsetPosition,
                    behavior: 'smooth'
                });
            }
        });
    });
}

// ===== Number Counter Animation =====
function initNumberCounter() {
    const counters = document.querySelectorAll('[data-counter]');
    
    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(function(entry) {
            if (entry.isIntersecting) {
                const counter = entry.target;
                const target = parseInt(counter.dataset.counter);
                const duration = parseInt(counter.dataset.duration) || 2000;
                
                animateCounter(counter, target, duration);
                observer.unobserve(counter);
            }
        });
    }, { threshold: 0.5 });
    
    counters.forEach(function(counter) {
        observer.observe(counter);
    });
}

function animateCounter(element, target, duration) {
    let start = 0;
    const increment = target / (duration / 16);
    
    const timer = setInterval(function() {
        start += increment;
        
        if (start >= target) {
            element.textContent = target.toLocaleString();
            clearInterval(timer);
        } else {
            element.textContent = Math.floor(start).toLocaleString();
        }
    }, 16);
}

// ===== Score Meter Animation =====
function animateScoreMeter(score) {
    const meter = document.querySelector('.score-meter-progress');
    if (meter) {
        const circumference = 565;
        const offset = circumference - (score / 100) * circumference;
        meter.style.strokeDashoffset = offset;
    }
}

// ===== Form Auto-formatting =====
function formatCurrency(input) {
    let value = input.value.replace(/[^0-9]/g, '');
    value = (parseInt(value) / 100).toFixed(2);
    input.value = '$' + value.replace(/\B(?=(\d{3})+(?!\d))/g, ',');
}

function formatPhoneNumber(input) {
    let value = input.value.replace(/\D/g, '');
    if (value.length >= 10) {
        input.value = '(' + value.slice(0, 3) + ') ' + value.slice(3, 6) + '-' + value.slice(6, 10);
    }
}

// ===== Toast Notifications =====
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `
        <div class="toast-header">
            <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle'}"></i>
            <strong class="me-auto">${type.charAt(0).toUpperCase() + type.slice(1)}</strong>
            <button type="button" class="btn-close" data-bs-dismiss="toast"></button>
        </div>
        <div class="toast-body">${message}</div>
    `;
    
    const container = document.querySelector('.toast-container') || document.body;
    container.appendChild(toast);
    
    const bsToast = new bootstrap.Toast(toast, { delay: 5000 });
    bsToast.show();
    
    toast.addEventListener('hidden.bs.toast', function() {
        toast.remove();
    });
}

// ===== Modal Helpers =====
function openModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        const bsModal = new bootstrap.Modal(modal);
        bsModal.show();
    }
}

function closeModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        const bsModal = bootstrap.Modal.getInstance(modal);
        if (bsModal) {
            bsModal.hide();
        }
    }
}

// ===== Loading States =====
function showLoading(button) {
    const originalText = button.innerHTML;
    button.disabled = true;
    button.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Loading...';
    button.dataset.originalText = originalText;
}

function hideLoading(button) {
    button.disabled = false;
    button.innerHTML = button.dataset.originalText || button.innerHTML;
}

// ===== API Helpers =====
async function fetchData(url, options = {}) {
    try {
        const response = await fetch(url, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('Fetch error:', error);
        showToast('An error occurred. Please try again.', 'error');
        throw error;
    }
}

// ===== Local Storage Helpers =====
function saveToLocalStorage(key, value) {
    localStorage.setItem(key, JSON.stringify(value));
}

function getFromLocalStorage(key) {
    const item = localStorage.getItem(key);
    return item ? JSON.parse(item) : null;
}

function removeFromLocalStorage(key) {
    localStorage.removeItem(key);
}

// ===== Session Storage Helpers =====
function saveToSessionStorage(key, value) {
    sessionStorage.setItem(key, JSON.stringify(value));
}

function getFromSessionStorage(key) {
    const item = sessionStorage.getItem(key);
    return item ? JSON.parse(item) : null;
}

// ===== URL Parameter Helpers =====
function getUrlParameter(name) {
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get(name);
}

function setUrlParameter(name, value) {
    const url = new URL(window.location);
    url.searchParams.set(name, value);
    window.history.replaceState({}, '', url);
}

// ===== Debounce Function =====
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// ===== Throttle Function =====
function throttle(func, limit) {
    let inThrottle;
    return function(...args) {
        if (!inThrottle) {
            func.apply(this, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// ===== Export functions for global use =====
window.LoanAI = {
    showToast,
    openModal,
    closeModal,
    showLoading,
    hideLoading,
    fetchData,
    saveToLocalStorage,
    getFromLocalStorage,
    removeFromLocalStorage,
    saveToSessionStorage,
    getFromSessionStorage,
    getUrlParameter,
    setUrlParameter,
    animateScoreMeter,
    formatCurrency,
    formatPhoneNumber
};
