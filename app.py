import streamlit as st
import pandas as pd
import os
import nltk
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Download NLTK data at app startup
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

# Call the download function
download_nltk_data()

from analyzer import process_resume, GEMINI_AVAILABLE
from database import ResumeDatabase

# Initialize database
db = ResumeDatabase()

# Page configuration
st.set_page_config(page_title="AI Resume Matcher", layout="wide", initial_sidebar_state="collapsed")

# Advanced Professional CSS with Enhanced Background
st.markdown("""
    <style>
    /* Import Professional Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&family=Poppins:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    :root {
        --primary: #6366f1;  /* Modern Indigo */
        --primary-dark: #4f46e5;
        --secondary: #10b981;  /* Emerald */
        --accent: #f43f5e;     /* Rose */
        --warning: #f59e0b;   /* Amber */
        --dark: #0f172a;
        --light: #f8fafc;
        --gray: #64748b;
        --card-bg: rgba(255, 255, 255, 0.98);
        --card-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -6px rgba(0, 0, 0, 0.05);
        --gradient-1: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        --gradient-2: linear-gradient(135deg, #ec4899 0%, #f43f5e 100%);
        --gradient-3: linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%);
        --gradient-4: linear-gradient(135deg, #10b981 0%, #14b8a6 100%);
    }
    
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(rgba(15, 23, 42, 0.85), rgba(30, 41, 59, 0.9)), 
                    url('https://images.unsplash.com/photo-1486312338219-ce68d2c6f44d?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2072&q=80');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no-repeat;
        color: #000000;  /* Changed to black */
        line-height: 1.6;
        min-height: 100vh;
    }
    
    .stApp {
        background: transparent;
        padding: 0;
        margin: 0;
        overflow-x: hidden;
    }
    
    /* Animated Background Elements */
    .bg-animation {
        position: fixed;
        width: 100%;
        height: 100%;
        top: 0;
        left: 0;
        z-index: -1;
        overflow: hidden;
    }
    
    .bg-animation span {
        position: absolute;
        display: block;
        width: 20px;
        height: 20px;
        background: rgba(255, 255, 255, 0.05);
        animation: move 25s linear infinite;
        bottom: -150px;
        border-radius: 50%;
    }
    
    .bg-animation span:nth-child(1) {
        left: 25%;
        width: 80px;
        height: 80px;
        animation-delay: 0s;
    }
    
    .bg-animation span:nth-child(2) {
        left: 10%;
        width: 20px;
        height: 20px;
        animation-delay: 2s;
        animation-duration: 12s;
    }
    
    .bg-animation span:nth-child(3) {
        left: 70%;
        width: 20px;
        height: 20px;
        animation-delay: 4s;
    }
    
    .bg-animation span:nth-child(4) {
        left: 40%;
        width: 60px;
        height: 60px;
        animation-delay: 0s;
        animation-duration: 18s;
    }
    
    .bg-animation span:nth-child(5) {
        left: 65%;
        width: 20px;
        height: 20px;
        animation-delay: 0s;
    }
    
    .bg-animation span:nth-child(6) {
        left: 75%;
        width: 110px;
        height: 110px;
        animation-delay: 3s;
    }
    
    .bg-animation span:nth-child(7) {
        left: 35%;
        width: 150px;
        height: 150px;
        animation-delay: 7s;
    }
    
    .bg-animation span:nth-child(8) {
        left: 50%;
        width: 25px;
        height: 25px;
        animation-delay: 15s;
        animation-duration: 45s;
    }
    
    .bg-animation span:nth-child(9) {
        left: 20%;
        width: 15px;
        height: 15px;
        animation-delay: 2s;
        animation-duration: 35s;
    }
    
    .bg-animation span:nth-child(10) {
        left: 85%;
        width: 150px;
        height: 150px;
        animation-delay: 0s;
        animation-duration: 11s;
    }
    
    @keyframes move {
        0% {
            transform: translateY(0) rotate(0deg);
            opacity: 1;
        }
        100% {
            transform: translateY(-1000px) rotate(720deg);
            opacity: 0;
        }
    }
    
    /* Header and Navigation */
    .main-header {
        background: rgba(15, 23, 42, 0.7);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.5rem 2rem;
        margin-bottom: 2rem;
        border-radius: 0 0 24px 24px;
        box-shadow: var(--card-shadow);
        position: relative;
        z-index: 10;
    }
    
    .main-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.5rem;
        font-weight: 800;
        color: white;
        text-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        margin: 0;
        letter-spacing: -0.025em;
        background: var(--gradient-1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-fill-color: transparent;
        display: inline-block;
    }
    
    .nav-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 1.5rem;
        gap: 1rem;
    }
    
    .nav-button {
        background: var(--gradient-1);
        color: white;
        border: none;
        padding: 0.875rem 2rem;
        border-radius: 16px;
        font-weight: 600;
        font-size: 1rem;
        cursor: pointer;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 0 10px 15px -3px rgba(99, 102, 241, 0.3);
        position: relative;
        overflow: hidden;
        z-index: 1;
        width: 100%;
    }
    
    .nav-button:before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 0;
        height: 100%;
        background: rgba(255, 255, 255, 0.2);
        transition: width 0.4s ease-in-out;
        z-index: -1;
    }
    
    .nav-button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 25px -5px rgba(99, 102, 241, 0.4);
    }
    
    .nav-button:hover:before {
        width: 100%;
    }
    
    .nav-button:active {
        transform: translateY(-1px);
    }
    
    .nav-button.active {
        background: var(--gradient-2);
        box-shadow: 0 10px 15px -3px rgba(236, 72, 153, 0.3);
    }
    
    /* Unified Content Section */
    .content-section {
        background: rgba(255, 255, 255, 0.98);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 2.8rem;
        margin: 2.8rem 0;
        box-shadow: var(--card-shadow);
        border: 1px solid rgba(0, 0, 0, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .content-section:before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 5px;
        background: var(--gradient-3);
    }
    
    .section-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.75rem;
        font-weight: 700;
        color: #000000;
        margin-bottom: 1.8rem;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .section-icon {
        width: 48px;
        height: 48px;
        background: var(--gradient-1);
        color: white;
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        box-shadow: 0 10px 15px -3px rgba(99, 102, 241, 0.3);
    }
    
    /* Form Elements */
    .stTextArea, .stFileUploader, .stSelectbox, .stSlider {
        background: rgba(255, 255, 255, 0.98);
        border-radius: 16px;
        border: 1px solid rgba(0, 0, 0, 0.3);
        padding: 1.4rem;
        margin-bottom: 1.4rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        color: #000000;
    }
    
    .stTextArea:focus, .stFileUploader:focus, .stSelectbox:focus, .stSlider:focus {
        border-color: var(--primary);
        box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.1);
        outline: none;
    }
    
    .stTextArea > div > textarea {
        font-family: 'Poppins', sans-serif;
        font-size: 1rem;
        line-height: 1.5;
        color: #000000;
    }
    
    /* Buttons */
    .stButton > button {
        background: var(--gradient-1);
        color: white;
        border: none;
        padding: 1rem 2.5rem;
        border-radius: 16px;
        font-weight: 600;
        font-size: 1.1rem;
        cursor: pointer;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 0 10px 15px -3px rgba(99, 102, 241, 0.3);
        width: 100%;
        margin-top: 1.8rem;
        position: relative;
        overflow: hidden;
        z-index: 1;
    }
    
    .stButton > button:before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 0;
        height: 100%;
        background: rgba(255, 255, 255, 0.2);
        transition: width 0.4s ease-in-out;
        z-index: -1;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 25px -5px rgba(99, 102, 241, 0.4);
    }
    
    .stButton > button:hover:before {
        width: 100%;
    }
    
    .stButton > button:active {
        transform: translateY(-1px);
    }
    
    /* Progress Bar */
    .stProgress > div > div > div {
        background: var(--gradient-3);
        border-radius: 10px;
        height: 12px;
    }
    
    /* Metrics */
    .metric-container {
        display: flex;
        flex-wrap: wrap;
        gap: 2.2rem;
        margin: 2.2rem 0;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.98);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 24px;
        padding: 2.4rem;
        flex: 1;
        min-width: 200px;
        box-shadow: var(--card-shadow);
        border: 1px solid rgba(0, 0, 0, 0.1);
        text-align: center;
        transition: all 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
        margin: 1.8rem 0;
    }
    
    .metric-card:before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 5px;
        background: var(--gradient-4);
    }
    
    .metric-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 15px 25px -5px rgba(0, 0, 0, 0.2), 0 10px 10px -5px rgba(0, 0, 0, 0.1);
    }
    
    .metric-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3rem;
        font-weight: 800;
        color: #000000;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 1.1rem;
        color: #000000;
        font-weight: 600;
    }
    
    /* Data Display */
    .dataframe-container {
        background: rgba(255, 255, 255, 0.98);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 24px;
        padding: 2.4rem;
        margin: 2.4rem 0;
        box-shadow: var(--card-shadow);
        overflow: hidden;
        border: 1px solid rgba(0, 0, 0, 0.1);
    }
    
    .stDataFrame {
        border-radius: 16px;
        overflow: hidden;
    }
    
    .stDataFrame > div > div > table {
        border-collapse: collapse;
        width: 100%;
    }
    
    .stDataFrame > div > div > table th {
        background: var(--gradient-1);
        color: #000000;
        font-weight: 600;
        text-align: left;
        padding: 1.25rem;
        font-size: 1.1rem;
    }
    
    .stDataFrame > div > div > table td {
        padding: 1.25rem;
        border-bottom: 1px solid rgba(0, 0, 0, 0.1);
        font-size: 1rem;
        color: #000000;
    }
    
    /* Suggestions */
    .suggestions-container {
        background: rgba(255, 255, 255, 0.98);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 24px;
        padding: 2.4rem;
        margin: 2.4rem 0;
        box-shadow: var(--card-shadow);
        border: 1px solid rgba(0, 0, 0, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .suggestions-container:before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 5px;
        background: var(--gradient-2);
    }
    
    .suggestions-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.75rem;
        font-weight: 700;
        color: #000000;
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .suggestion-item {
        background: rgba(99, 102, 241, 0.1);
        border-left: 5px solid var(--primary);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border-radius: 0 16px 16px 0;
        font-size: 1.1rem;
        color: #000000;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .suggestion-item:hover {
        background: rgba(99, 102, 241, 0.2);
        transform: translateX(8px);
        box-shadow: 0 10px 15px -3px rgba(99, 102, 241, 0.2);
    }
    
    /* Verdict Badge */
    .verdict-badge {
        display: inline-block;
        padding: 0.75rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1.1rem;
        margin-top: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        color: #000000;
    }
    
    .verdict-high {
        background: rgba(16, 185, 129, 0.2);
        border: 1px solid rgba(16, 185, 129, 0.5);
    }
    
    .verdict-medium {
        background: rgba(236, 72, 153, 0.2);
        border: 1px solid rgba(236, 72, 153, 0.5);
    }
    
    .verdict-low {
        background: rgba(59, 130, 246, 0.2);
        border: 1px solid rgba(59, 130, 246, 0.5);
    }
    
    /* Hide Streamlit Elements */
    #MainMenu, header, footer, [data-testid="stDeployButton"], [data-testid="collapsedControl"] {
        visibility: hidden;
        height: 0;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2rem;
        }
        
        .nav-container {
            flex-direction: column;
            gap: 1rem;
        }
        
        .metric-container {
            flex-direction: column;
            gap: 1rem;
        }
        
        .content-section {
            padding: 1.8rem;
            margin: 1.8rem 0;
        }
        
        .section-title {
            font-size: 1.5rem;
        }
    }
    
    /* API Key Status */
    .api-status {
        background: rgba(16, 185, 129, 0.1);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(16, 185, 129, 0.2);
        display: flex;
        align-items: center;
        gap: 1rem;
        box-shadow: 0 10px 15px -3px rgba(16, 185, 129, 0.1);
    }
    
    .api-status-icon {
        width: 32px;
        height: 32px;
        background: var(--gradient-4);
        color: white;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
        box-shadow: 0 4px 6px -1px rgba(16, 185, 129, 0.3);
    }
    
    .api-status-text {
        font-size: 1.1rem;
        color: #000000;
        font-weight: 600;
    }
    
    .api-error {
        background: rgba(244, 63, 94, 0.1);
        border: 1px solid rgba(244, 63, 94, 0.2);
    }
    
    .api-error .api-status-icon {
        background: var(--gradient-2);
    }
    
    .api-error .api-status-text {
        color: #000000;
    }
    
    /* Error Message */
    .error-container {
        background: rgba(244, 63, 94, 0.1);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 24px;
        padding: 2.5rem;
        margin: 2rem 0;
        box-shadow: var(--card-shadow);
        border: 1px solid rgba(244, 63, 94, 0.2);
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .error-container:before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 5px;
        background: var(--gradient-3);
    }
    
    .error-icon {
        font-size: 4rem;
        color: var(--accent);
        margin-bottom: 1.5rem;
    }
    
    .error-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.75rem;
        font-weight: 700;
        color: #000000;
        margin-bottom: 1.5rem;
    }
    
    .error-message {
        font-size: 1.1rem;
        color: #000000;
        margin-bottom: 2rem;
        line-height: 1.6;
    }
    
    /* Loading Animation */
    .loading-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 3rem;
    }
    
    .loading-spinner {
        width: 60px;
        height: 60px;
        border: 5px solid rgba(255, 255, 255, 0.3);
        border-radius: 50%;
        border-top-color: var(--primary);
        animation: spin 1s ease-in-out infinite;
        margin-bottom: 1.5rem;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    .loading-text {
        font-size: 1.2rem;
        color: #000000;
        font-weight: 500;
    }
    
    /* Fix Streamlit Elements */
    .element-container {
        margin-bottom: 1rem;
    }
    
    /* Fix for text area height */
    .stTextArea > div > div > textarea {
        min-height: 220px;
    }
    
    /* Fix for file uploader */
    .stFileUploader > div > div > span {
        font-weight: 500;
        color: #000000;
    }
    
    /* Fix for selectbox */
    .stSelectbox > div > div > select {
        font-weight: 500;
        color: #000000;
    }
    
    /* Fix for slider */
    .stSlider > div > div > div > div {
        background: var(--gradient-1);
    }
    
    /* Fix for dataframe */
    .stDataFrame {
        border: none;
    }
    
    /* Fix for success and error messages */
    .stSuccess, .stError, .stInfo, .stWarning {
        border-radius: 16px;
        padding: 1rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        color: #000000;
    }
    
    .stSuccess {
        background: rgba(16, 185, 129, 0.2);
        border-left: 5px solid var(--secondary);
    }
    
    .stError {
        background: rgba(244, 63, 94, 0.2);
        border-left: 5px solid var(--accent);
    }
    
    .stInfo {
        background: rgba(99, 102, 241, 0.2);
        border-left: 5px solid var(--primary);
    }
    
    .stWarning {
        background: rgba(245, 158, 11, 0.2);
        border-left: 5px solid var(--warning);
    }
    
    /* Text Color Adjustments */
    h1, h2, h3, h4, h5, h6, .stMarkdown, p, span, div {
        color: #000000;
    }
    
    .stTextInput > div > input {
        color: #000000;
    }
    
    /* Dashboard specific styles */
    .stSelectbox[data-testid="stSelectbox"] label {
        color: #000000;
        font-weight: 600;
    }
    
    .stSlider[data-testid="stSlider"] label {
        color: #000000;
        font-weight: 600;
    }
    
    /* Evaluation details styles */
    .evaluation-details {
        background: rgba(255, 255, 255, 0.98);
        border-radius: 24px;
        padding: 2.4rem;
        margin: 2.4rem 0;
        box-shadow: var(--card-shadow);
        border: 1px solid rgba(0, 0, 0, 0.1);
    }
    
    .evaluation-details h3 {
        color: #000000;
        margin-bottom: 1.5rem;
    }
    
    .evaluation-details p {
        color: #000000;
        margin-bottom: 1.2rem;
    }
    
    /* Animated Background Elements */
    .bg-animation {
        position: fixed;
        width: 100%;
        height: 100%;
        top: 0;
        left: 0;
        z-index: -1;
        overflow: hidden;
    }
    
    .bg-animation span {
        position: absolute;
        display: block;
        width: 20px;
        height: 20px;
        background: rgba(255, 255, 255, 0.05);
        animation: move 25s linear infinite;
        bottom: -150px;
        border-radius: 50%;
    }
    
    .bg-animation span:nth-child(1) {
        left: 25%;
        width: 80px;
        height: 80px;
        animation-delay: 0s;
    }
    
    .bg-animation span:nth-child(2) {
        left: 10%;
        width: 20px;
        height: 20px;
        animation-delay: 2s;
        animation-duration: 12s;
    }
    
    .bg-animation span:nth-child(3) {
        left: 70%;
        width: 20px;
        height: 20px;
        animation-delay: 4s;
    }
    
    .bg-animation span:nth-child(4) {
        left: 40%;
        width: 60px;
        height: 60px;
        animation-delay: 0s;
        animation-duration: 18s;
    }
    
    .bg-animation span:nth-child(5) {
        left: 65%;
        width: 20px;
        height: 20px;
        animation-delay: 0s;
    }
    
    .bg-animation span:nth-child(6) {
        left: 75%;
        width: 110px;
        height: 110px;
        animation-delay: 3s;
    }
    
    .bg-animation span:nth-child(7) {
        left: 35%;
        width: 150px;
        height: 150px;
        animation-delay: 7s;
    }
    
    .bg-animation span:nth-child(8) {
        left: 50%;
        width: 25px;
        height: 25px;
        animation-delay: 15s;
        animation-duration: 45s;
    }
    
    .bg-animation span:nth-child(9) {
        left: 20%;
        width: 15px;
        height: 15px;
        animation-delay: 2s;
        animation-duration: 35s;
    }
    
    .bg-animation span:nth-child(10) {
        left: 85%;
        width: 150px;
        height: 150px;
        animation-delay: 0s;
        animation-duration: 11s;
    }
    
    @keyframes move {
        0% {
            transform: translateY(0) rotate(0deg);
            opacity: 1;
        }
        100% {
            transform: translateY(-1000px) rotate(720deg);
            opacity: 0;
        }
    }
    </style>
    
    <!-- Animated Background Elements -->
    <div class="bg-animation">
        <span></span>
        <span></span>
        <span></span>
        <span></span>
        <span></span>
        <span></span>
        <span></span>
        <span></span>
        <span></span>
        <span></span>
    </div>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if "page" not in st.session_state:
        st.session_state.page = "home"
    if "analyzed" not in st.session_state:
        st.session_state.analyzed = False
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None

# Call the initialization function
init_session_state()

# Get Gemini API key from environment variable
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Navigation
def navigation():
    st.markdown("""
        <div class="main-header">
            <h1 class="main-title">Automated Resume Relevance Check System</h1>
            <div class="nav-container">
    """, unsafe_allow_html=True)
    
    # Navigation buttons using Streamlit
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🏠 Home", key="nav_home", help="Go to Home Page"):
            st.session_state.page = "home"
            st.query_params["page"] = "home"
            st.rerun()
    
    with col2:
        if st.button("📊 Dashboard", key="nav_dashboard", help="Go to Dashboard"):
            st.session_state.page = "dashboard"
            st.query_params["page"] = "dashboard"
            st.rerun()
    
    st.markdown("""
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Handle navigation via query params
    if "page" in st.query_params:
        st.session_state.page = st.query_params["page"]

# Home page
def show_home():
    # Show API key status
    if gemini_api_key:
        st.markdown("""
            <div class="api-status">
                <div class="api-status-icon">✓</div>
                <div class="api-status-text">Gemini API key is configured and ready to use</div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="api-status api-error">
                <div class="api-status-icon">!</div>
                <div class="api-status-text">Gemini API key is not configured. Please add it to your .env file.</div>
            </div>
        """, unsafe_allow_html=True)
        st.error("""
            **How to set up your Gemini API key:**
            
            1. Create a `.env` file in your project root directory
            2. Add the following line to the file:
               ```
               GEMINI_API_KEY=your_gemini_api_key_here
               ```
            3. Replace `your_gemini_api_key_here` with your actual API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
            4. Restart the application
        """)
    
    st.markdown("""
        <div class="content-section">
            <div class="section-title">
                <div class="section-icon">📝</div>
                Job Description
            </div>
            <p style="color: #000000; margin-bottom: 1.2rem;">
                Enter the complete job description including required skills, qualifications, and responsibilities.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    job_desc = st.text_area("Paste the job description here", height=220, 
                            placeholder="Enter the complete job description including required skills, qualifications, and responsibilities...")
    
    st.markdown("""
        <div class="content-section">
            <div class="section-title">
                <div class="section-icon">📄</div>
                Upload Resume
            </div>
            <p style="color: #000000; margin-bottom: 1.2rem;">
                Upload your resume in PDF or DOCX format for analysis.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload your resume (PDF or DOCX)", type=["pdf", "docx"])
    
    # Show Gemini availability status
    if not GEMINI_AVAILABLE:
        st.error("⚠️ Gemini integration not available. Install google-generativeai package.")
    
    if uploaded_file and job_desc:
        if not gemini_api_key:
            st.error("🔑 Please configure your Gemini API key in the .env file.")
        elif not GEMINI_AVAILABLE:
            st.error("⚠️ Gemini integration is not available. Please install the required packages.")
        else:
            if st.button("🚀 Analyze Resume", key="analyze_button"):
                with st.spinner("Analyzing your resume with Gemini AI..."):
                    # Save the uploaded file temporarily
                    temp_dir = "temp"
                    os.makedirs(temp_dir, exist_ok=True)
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    
                    # Initialize Gemini analyzer
                    from gemini_integration import GeminiAnalyzer
                    gemini_analyzer = GeminiAnalyzer(api_key=gemini_api_key)
                    
                    # Process the resume
                    results = process_resume(file_path, job_desc, gemini_analyzer=gemini_analyzer)
                    
                    # Save to database
                    db.save_evaluation(uploaded_file.name, job_desc, results)
                    
                    # Store results in session state
                    st.session_state.analysis_results = results
                    st.session_state.analyzed = True
                    st.session_state.page = "results"
                    st.query_params["page"] = "results"
                    st.rerun()
    else:
        st.info("📥 Please upload a resume and enter the job description to begin analysis.")

# Results page
def show_results():
    # Check if analysis results exist
    if st.session_state.analysis_results is None:
        st.markdown("""
            <div class="error-container">
                <div class="error-icon">⚠️</div>
                <div class="error-title">No Analysis Results Found</div>
                <div class="error-message">
                    It looks like you haven't analyzed any resume yet. Please go back to the home page to upload a resume and job description.
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔙 Back to Home", key="no_results_back_button"):
            st.session_state.page = "home"
            st.query_params["page"] = "home"
            st.rerun()
        return
    
    results = st.session_state.analysis_results
    
    st.markdown("""
        <div class="content-section">
            <div class="section-title">
                <div class="section-icon">📊</div>
                Resume Match Results
            </div>
            <p style="color: #000000; margin-bottom: 1.2rem;">
                Your resume has been analyzed against the job description. Here are the results:
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Score and verdict
    st.markdown("""
        <div class="metric-container">
            <div class="metric-card">
                <div class="metric-label">Matching Score</div>
                <div class="metric-value">{}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Verdict</div>
                <div class="verdict-badge verdict-{}">{}</div>
            </div>
        </div>
    """.format(results['score'], results['verdict'].lower(), results['verdict']), unsafe_allow_html=True)
    
    # Progress bar
    st.progress(results["score"] / 100)
    
    # Score breakdown
    st.markdown("""
        <div class="content-section">
            <div class="section-title">
                <div class="section-icon">📈</div>
                Score Breakdown
            </div>
            <p style="color: #000000; margin-bottom: 1.2rem;">
                Detailed breakdown of how your resume matches the job requirements:
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Hard Match Score</div>
                <div class="metric-value">{}%</div>
            </div>
        """.format(results['hard_score']), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Semantic Match Score</div>
                <div class="metric-value">{}%</div>
            </div>
        """.format(results['semantic_score']), unsafe_allow_html=True)
    
    # Skills analysis
    st.markdown("""
        <div class="content-section">
            <div class="section-title">
                <div class="section-icon">✅</div>
                Matched Skills
            </div>
            <p style="color: #000000; margin-bottom: 1.2rem;">
                Skills from your resume that match the job requirements:
            </p>
        </div>
    """, unsafe_allow_html=True)
    st.success(", ".join(results["matched_skills"]))
    
    st.markdown("""
        <div class="content-section">
            <div class="section-title">
                <div class="section-icon">⚠️</div>
                Missing Skills
            </div>
            <p style="color: #000000; margin-bottom: 1.2rem;">
                Skills mentioned in the job description that are not present in your resume:
            </p>
        </div>
    """, unsafe_allow_html=True)
    st.error(", ".join(results["missing_skills"]))
    
    # Suggestions
    st.markdown("""
        <div class="suggestions-container">
            <div class="suggestions-header">
                <div class="section-icon">💡</div>
                Suggestions to Improve Resume
            </div>
            <p style="color: #000000; margin-bottom: 1.5rem;">
                Here are some suggestions to improve your resume based on the analysis:
            </p>
    """, unsafe_allow_html=True)
    
    for suggestion in results["suggestions"]:
        st.markdown(f'<div class="suggestion-item">🔹 {suggestion}</div>', unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Priority table
    st.markdown("""
        <div class="content-section">
            <div class="section-title">
                <div class="section-icon">📌</div>
                Priority Improvements
            </div>
            <p style="color: #000000; margin-bottom: 1.2rem;">
                Focus on these high-priority improvements to increase your match score:
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    df = pd.DataFrame(results["priority_table"])
    st.dataframe(df, use_container_width=True)
    
    # Back button
    if st.button("🔙 Back to Home", key="back_button"):
        st.session_state.page = "home"
        st.query_params["page"] = "home"
        st.rerun()

# Dashboard page
def show_dashboard():
    st.markdown("""
        <div class="content-section">
            <div class="section-title">
                <div class="section-icon">📊</div>
                Placement Team Dashboard
            </div>
            <p style="color: #000000; margin-bottom: 1.2rem;">
                View and manage all resume evaluations:
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Filters
    st.markdown("""
        <div class="content-section">
            <div class="section-title">
                <div class="section-icon">🔍</div>
                Filter Evaluations
            </div>
            <p style="color: #000000; margin-bottom: 1.2rem;">
                Filter evaluations by score and verdict:
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        min_score = st.slider("Minimum Score", 0, 100, 0)
    with col2:
        verdict_filter = st.selectbox("Verdict", ["All", "High", "Medium", "Low"])
    
    # Apply filters
    filters = {"min_score": min_score}
    if verdict_filter != "All":
        filters["verdict"] = verdict_filter
    
    # Get evaluations
    evaluations = db.get_evaluations(filters)
    
    # Display evaluations
    st.markdown("""
        <div class="content-section">
            <div class="section-title">
                <div class="section-icon">📋</div>
                Resume Evaluations ({})
            </div>
            <p style="color: #000000; margin-bottom: 1.2rem;">
                List of all resume evaluations matching your filters:
            </p>
        </div>
    """.format(len(evaluations)), unsafe_allow_html=True)
    
    if evaluations:
        # Create a dataframe for display
        data = []
        for eval in evaluations:
            data.append({
                "ID": eval["id"],
                "Resume": eval["resume_name"],
                "Score": eval["score"],
                "Verdict": eval["verdict"],
                "Date": eval["evaluation_date"]
            })
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
        
        # Allow viewing details
        selected_id = st.selectbox("Select evaluation to view details", df["ID"])
        if selected_id:
            selected_eval = next(eval for eval in evaluations if eval["id"] == selected_id)
            display_evaluation_details(selected_eval)
    else:
        st.info("No evaluations found matching the selected criteria.")

def display_evaluation_details(evaluation):
    st.markdown("""
        <div class="evaluation-details">
            <div class="section-title">
                <div class="section-icon">📝</div>
                Evaluation Details
            </div>
            <p style="color: #000000; margin-bottom: 1.2rem;">
                Detailed analysis results for this resume:
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Score</div>
                <div class="metric-value">{}%</div>
            </div>
        """.format(evaluation['score']), unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Verdict</div>
                <div class="verdict-badge verdict-{}">{}</div>
            </div>
        """.format(evaluation['verdict'].lower(), evaluation['verdict']), unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Hard Match</div>
                <div class="metric-value">{}%</div>
            </div>
        """.format(evaluation['hard_score']), unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Semantic Match</div>
                <div class="metric-value">{}%</div>
            </div>
        """.format(evaluation['semantic_score']), unsafe_allow_html=True)
    
    st.markdown("""
        <div class="content-section">
            <div class="section-title">
                <div class="section-icon">✅</div>
                Matched Skills
            </div>
            <p style="color: #000000; margin-bottom: 1.2rem;">
                Skills from the resume that match the job requirements:
            </p>
        </div>
    """, unsafe_allow_html=True)
    st.success(", ".join(evaluation["matched_skills"].split(',')))
    
    st.markdown("""
        <div class="content-section">
            <div class="section-title">
                <div class="section-icon">⚠️</div>
                Missing Skills
            </div>
            <p style="color: #000000; margin-bottom: 1.2rem;">
                Skills mentioned in the job description that are not present in the resume:
            </p>
        </div>
    """, unsafe_allow_html=True)
    st.error(", ".join(evaluation["missing_skills"].split(',')))
    
    st.markdown("""
        <div class="suggestions-container">
            <div class="suggestions-header">
                <div class="section-icon">💡</div>
                Suggestions
            </div>
            <p style="color: #000000; margin-bottom: 1.5rem;">
                Suggestions to improve this resume:
            </p>
    """, unsafe_allow_html=True)
    
    suggestions = evaluation["suggestions"].split('|')
    for suggestion in suggestions:
        st.markdown(f'<div class="suggestion-item">🔹 {suggestion}</div>', unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Main app
def main():
    navigation()
    
    if st.session_state.page == "home":
        show_home()
    elif st.session_state.page == "results":
        show_results()
    elif st.session_state.page == "dashboard":
        show_dashboard()

if __name__ == "__main__":
    main()