from flask import Flask, request, jsonify
from dotenv import load_dotenv
from flask_cors import CORS 
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import List, Dict
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure GenAI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize Flask
app = Flask(__name__)
CORS(app)

# Pydantic Models for structured output
class CandidateSkillAnalysis(BaseModel):
    skill_name: str = Field(description="Name of the skill from the resume")
    relevance: str = Field(description="How relevant this skill is to the job")
    match_score: int = Field(description="Score (0-10) for how well this skill matches job requirements")

class CandidateAnalysis(BaseModel):
    candidate_id: str = Field(description="Identifier for the candidate")
    candidate_name: str = Field(description="Name of the candidate from resume", default="")
    overall_score: int = Field(description="Overall match score (0-100) for this candidate")
    strengths: List[str] = Field(description="List of strengths matching the job requirements")
    weaknesses: List[str] = Field(description="List of weaknesses or missing qualifications")
    skill_breakdown: List[CandidateSkillAnalysis] = Field(description="Detailed analysis of skills")
    recommendation: str = Field(description="Recommendation: Strong, Moderate, or Weak fit")
    reasons: List[str] = Field(description="Detailed reasons for the recommendation")

class RecruitmentAnalysis(BaseModel):
    job_description: str = Field(description="The provided job description")
    total_candidates: int = Field(description="Total number of candidates analyzed")
    openings_available: int = Field(description="Number of positions available")
    selected_candidates: List[str] = Field(description="IDs of candidates selected based on openings")
    rejected_candidates: List[str] = Field(description="IDs of candidates not selected")
    top_candidates: List[Dict] = Field(description="List of top candidates with their scores")
    candidate_analyses: List[CandidateAnalysis] = Field(description="Detailed analysis of all candidates")
    summary: str = Field(description="Summary of the recruitment analysis")
    final_verdict: str = Field(description="Final selection statement based on openings")

# Helper functions
def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def split_text_into_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_text(text)

def create_vector_store(text_chunks, store_name="faiss_index"):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(store_name)
    return vector_store

def get_analysis_chain():
    parser = PydanticOutputParser(pydantic_object=RecruitmentAnalysis)
    
    prompt_template = PromptTemplate(
        template="""Analyze these candidates for the given job opening. Consider there are {openings} positions available.
        
        Job Description:
        {job_description}
        
        Candidate Resumes:
        {resumes}
        
        Provide a detailed analysis including:
        1. Overall match scores for each candidate (0-100 scale)
        2. Key strengths and weaknesses relative to the job
        3. Skill-by-skill breakdown
        4. Clear recommendations on who to shortlist
        5. Reasons for each recommendation
        6. Final verdict explicitly stating how many candidates are selected based on the openings
        7. List of selected candidate IDs and rejected candidate IDs
        
        Important: The number of selected candidates must exactly match the number of openings specified.
        If there are more qualified candidates than openings, select the top ones.
        If there are fewer qualified candidates than openings, select all qualified ones.
        
        {format_instructions}""",
        input_variables=["job_description", "resumes", "openings"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.2)
    chain = prompt_template | model | parser
    return chain

def analyze_candidates(job_description, resumes_text, openings):
    try:
        chain = get_analysis_chain()
        response = chain.invoke({
            "job_description": job_description,
            "resumes": resumes_text,
            "openings": openings
        })
        
        # Ensure the number of selected candidates matches openings (as a fallback)
        if len(response.selected_candidates) > openings:
            response.selected_candidates = response.selected_candidates[:openings]
            response.rejected_candidates = response.selected_candidates[openings:] + response.rejected_candidates
            response.final_verdict = f"Selected top {openings} candidates: {', '.join(response.selected_candidates)}"
        
        return response
    except Exception as e:
        return {"error": str(e)}

def extract_candidate_name(resume_text):
    # Simple heuristic to extract name (first two lines are often name)
    lines = [line.strip() for line in resume_text.split('\n') if line.strip()]
    return lines[0] if lines else ""

# API Endpoints
@app.route('/api/analyze-resumes', methods=['POST'])
def analyze_resumes():
    if 'files' not in request.files:
        return jsonify({"error": "No files uploaded"}), 400
    
    files = request.files.getlist('files')
    job_description = request.form.get('job_description', '')
    openings = request.form.get('openings', 1)
    
    if not job_description:
        return jsonify({"error": "Job description is required"}), 400
    
    try:
        openings = int(openings)
    except ValueError:
        return jsonify({"error": "Number of openings must be an integer"}), 400
    
    # Process all resumes
    resumes_data = []
    for i, file in enumerate(files):
        if file.filename.endswith('.pdf'):
            text = extract_text_from_pdf(file)
            candidate_name = extract_candidate_name(text)
            resumes_data.append({
                "id": f"candidate_{i+1}",
                "name": candidate_name,
                "text": text,
                "filename": file.filename
            })
    
    if not resumes_data:
        return jsonify({"error": "No valid PDF files found"}), 400
    
    # Prepare input for analysis
    resumes_text = "\n\n".join([f"Candidate {i+1} (ID: {item['id']}, Name: {item['name']}):\n{item['text']}" 
                              for i, item in enumerate(resumes_data)])
    
    # Get analysis from LLM
    analysis = analyze_candidates(job_description, resumes_text, openings)
    
    if isinstance(analysis, dict) and 'error' in analysis:
        return jsonify(analysis), 500
    
    # Convert Pydantic model to dict
    response_data = analysis.dict()
    print(response_data)
    
    # Add some additional processing for better presentation
    response_data["openings_available"] = openings
    if openings == 1:
        response_data["final_verdict"] = f"Selected the top candidate: {analysis.selected_candidates[0] if analysis.selected_candidates else 'None'}"
    else:
        response_data["final_verdict"] = f"Selected {len(analysis.selected_candidates)} candidates for {openings} openings: {', '.join(analysis.selected_candidates)}"
    
    return jsonify(response_data)

@app.route('/api/test', methods=['GET'])
def test_endpoint():
    return jsonify({"message": "API is working"})

if __name__ == '__main__':
    app.run(port=5027, debug=True)



