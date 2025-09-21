import os
import pdfplumber
import docx2txt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from utils import clean_text, extract_keywords, tokenize_and_remove_stopwords

# Try importing Gemini analyzer, but make it optional
try:
    from gemini_integration import GeminiAnalyzer
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: Gemini integration not available. Install google-generativeai.")

def extract_text_from_pdf(file_path):
    """Extract text from PDF file"""
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(file_path):
    """Extract text from DOCX file"""
    return docx2txt.process(file_path)

def extract_text_from_resume(file_path):
    """Extract text from resume (PDF or DOCX)"""
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    else:
        return ""

def parse_job_description(job_desc):
    """Parse job description to extract key requirements"""
    # Clean the job description
    cleaned_desc = clean_text(job_desc)
    
    # Extract keywords (top 20)
    keywords = extract_keywords(cleaned_desc, n=20)
    
    return {
        "raw_text": job_desc,
        "cleaned_text": cleaned_desc,
        "keywords": keywords
    }

def calculate_hard_match_score(resume_text, job_keywords):
    """Calculate hard match score based on keyword presence"""
    # Clean and tokenize resume
    cleaned_resume = clean_text(resume_text)
    resume_tokens = set(tokenize_and_remove_stopwords(cleaned_resume))
    job_tokens = set(job_keywords)
    
    # Count how many job keywords are in the resume
    matched = len(resume_tokens.intersection(job_tokens))
    total = len(job_tokens)
    
    if total == 0:
        return 0
    
    return (matched / total) * 100

def calculate_semantic_match_score(resume_text, job_text, gemini_analyzer=None):
    """Calculate semantic match score using Gemini or TF-IDF"""
    if gemini_analyzer and GEMINI_AVAILABLE:
        # Use Gemini-based semantic analysis
        return gemini_analyzer.calculate_semantic_similarity(resume_text, job_text)
    else:
        # Fallback to TF-IDF
        texts = [resume_text, job_text]
        vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            return similarity_matrix[0][0] * 100
        except Exception as e:
            print(f"Error calculating TF-IDF similarity: {e}")
            # Fallback to simple word overlap
            resume_words = set(resume_text.lower().split())
            job_words = set(job_text.lower().split())
            if len(job_words) == 0:
                return 0
            overlap = len(resume_words.intersection(job_words))
            return (overlap / len(job_words)) * 100

def calculate_relevance_score(resume_text, job_data, gemini_analyzer=None):
    """Calculate overall relevance score (hybrid)"""
    # Hard match score (50% weight)
    hard_score = calculate_hard_match_score(resume_text, job_data["keywords"])
    
    # Semantic match score (50% weight)
    semantic_score = calculate_semantic_match_score(
        resume_text, job_data["cleaned_text"], gemini_analyzer
    )
    
    # Weighted average
    total_score = (0.5 * hard_score) + (0.5 * semantic_score)
    
    return {
        "score": round(total_score, 2),
        "hard_score": round(hard_score, 2),
        "semantic_score": round(semantic_score, 2)
    }

def identify_gaps(resume_text, job_keywords):
    """Identify missing skills/keywords in the resume"""
    cleaned_resume = clean_text(resume_text)
    resume_tokens = set(tokenize_and_remove_stopwords(cleaned_resume))
    job_tokens = set(job_keywords)
    
    missing = list(job_tokens - resume_tokens)
    matched = list(job_tokens.intersection(resume_tokens))
    
    return matched, missing

def generate_suggestions(missing_skills, resume_text, job_desc, gemini_analyzer=None):
    """Generate suggestions based on missing skills"""
    if gemini_analyzer and GEMINI_AVAILABLE:
        # Use Gemini to generate personalized suggestions
        return gemini_analyzer.generate_suggestions(resume_text, job_desc, missing_skills)
    else:
        # Fallback to generic suggestions
        suggestions = []
        for skill in missing_skills:
            suggestions.append(f"Consider adding experience with {skill}.")
        
        if len(missing_skills) > 5:
            suggestions.append("Your resume appears to be missing several key skills. Consider tailoring it more to the job description.")
        
        return suggestions

def determine_verdict(score):
    """Determine verdict based on score"""
    if score >= 80:
        return "High"
    elif score >= 60:
        return "Medium"
    else:
        return "Low"

def process_resume(resume_path, job_desc, gemini_analyzer=None):
    """Main function to process a resume against a job description"""
    # Extract text from resume
    resume_text = extract_text_from_resume(resume_path)
    
    # Parse job description
    job_data = parse_job_description(job_desc)
    
    # Calculate relevance score
    score_data = calculate_relevance_score(resume_text, job_data, gemini_analyzer)
    
    # Identify matched and missing skills
    matched_skills, missing_skills = identify_gaps(resume_text, job_data["keywords"])
    
    # Generate suggestions
    suggestions = generate_suggestions(
        missing_skills, resume_text, job_data["raw_text"], gemini_analyzer
    )
    
    # Determine verdict
    verdict = determine_verdict(score_data["score"])
    
    # Create priority table
    priority_table = []
    top_missing = missing_skills[:5]  # Top 5 missing skills
    for skill in top_missing:
        priority_table.append({
            "Improvement Suggestion": f"Add experience with {skill}.",
            "Priority": "🔴 High"
        })
    
    # Add medium priority for the rest
    for skill in missing_skills[5:]:
        priority_table.append({
            "Improvement Suggestion": f"Consider adding experience with {skill}.",
            "Priority": "🟡 Medium"
        })
    
    return {
        "score": score_data["score"],
        "hard_score": score_data["hard_score"],
        "semantic_score": score_data["semantic_score"],
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "suggestions": suggestions,
        "verdict": verdict,
        "priority_table": priority_table
    }