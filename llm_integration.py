from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class LLMAnalyzer:
    def __init__(self, openai_api_key=None):
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    
    def calculate_semantic_similarity(self, resume_text, job_desc):
        """Calculate semantic similarity between resume and job description"""
        # Create embeddings for both texts
        resume_embedding = np.array(self.embeddings.embed_query(resume_text))
        job_embedding = np.array(self.embeddings.embed_query(job_desc))
        
        # Reshape for cosine similarity
        resume_embedding = resume_embedding.reshape(1, -1)
        job_embedding = job_embedding.reshape(1, -1)
        
        # Calculate cosine similarity
        similarity = cosine_similarity(resume_embedding, job_embedding)[0][0]
        
        return similarity * 100  # Convert to percentage
    
    def generate_suggestions(self, resume_text, job_desc, missing_skills):
        """Generate personalized suggestions using LLM"""
        from langchain_community.llms import OpenAI
        
        llm = OpenAI(temperature=0, openai_api_key=self.embeddings.openai_api_key)
        
        prompt = f"""
        Based on the following resume and job description, provide specific suggestions 
        to improve the resume for this position:
        
        Resume: {resume_text[:2000]}...
        
        Job Description: {job_desc[:2000]}...
        
        Missing Skills: {', '.join(missing_skills)}
        
        Please provide 5-7 specific, actionable suggestions to improve this resume.
        """
        
        response = llm(prompt)
        return [s.strip() for s in response.split('\n') if s.strip()]