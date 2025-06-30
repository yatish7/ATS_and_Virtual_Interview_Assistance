from flask import Flask, request, jsonify
from dotenv import load_dotenv
from flask_cors import CORS 
import time
import os
from PyPDF2 import PdfReader
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from typing import TypedDict, Annotated
from pydantic import BaseModel, Field
import google.generativeai as genai
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document
from typing import List, Dict, Any, Optional

# Load environment variables
load_dotenv()

# Configure GenAI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash-latest")

# Initialize Flask
app = Flask(__name__)
CORS(app)

# Resume Analysis Pydantic Models
class SkillAnalysis(BaseModel):
    skill_name: str = Field(description="Name of the skill.")
    match_percentage: int = Field(description="Match percentage of the skill based on job description.")

class ResumeAnalysis(BaseModel):
    weaker_sections: list[str] = Field(description="List of weaker areas in the resume.")
    medium_sections: list[str] = Field(description="List of areas in the resume that are average.")
    strong_sections: list[str] = Field(description="List of strong areas in the resume.")
    overall_score: int = Field(description="Overall score for the resume on a scale of 10.")
    suggestions: list[str] = Field(description="Suggestions to improve the resume.")
    verdict: str = Field(description="Tell whether the candidate is fit for the role according to the job description provided.")
    skill_analysis: list[SkillAnalysis] = Field(description="Detailed analysis of skills and their match percentages.")

# Function to extract text from PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_text(text)

# Function to create vector store
def get_vector_store(text_chunks, vector_store_path):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(vector_store_path)
    return vector_store

# Function to load vector store
def load_vector_store(vector_store_path):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)

# Function to set up conversational chain
def get_conversational_chain():
    # Define the parser
    parser = PydanticOutputParser(pydantic_object=ResumeAnalysis)
    
    # Define the prompt template
    prompt_template = PromptTemplate(
        template=(
            "Analyze the following resume and categorize the content into weaker sections, "
            "medium sections, and strong sections against a particular job description that would be provided to you, "
            "and give a verdict whether he/she is fit for that particular role. And also provide overall score for the resume. "
            "Provide actionable suggestions for improvement.\n\n"
            "Give response in the point of a fresher who has just passed from college with no professional experience.\n"
            "{format_instructions}\n\nResume:\n{context}\n\nJob Description:\n{job}\n"
        ),
        input_variables=["context", "job"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # Set up the chain
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3)
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt_template)
    return chain, parser

# Evaluation prompts
relevance_instructions = """You are a teacher grading a resume analysis. 

You will be given a RESUME, a JOB DESCRIPTION, and an ANALYSIS of the resume. 

Here is the grade criteria to follow:
(1) Ensure the ANALYSIS is concise and relevant to both the RESUME and the JOB DESCRIPTION
(2) Ensure the ANALYSIS properly evaluates the resume against the job description
(3) Ensure the ANALYSIS provides meaningful insights about the candidate's fit for the position

Relevance:
A relevance value of True means that the analysis meets all of the criteria.
A relevance value of False means that the analysis does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""

groundedness_instructions = """You are a teacher grading a resume analysis. 

You will be given a RESUME and an ANALYSIS of that resume. 

Here is the grade criteria to follow:
(1) Ensure the ANALYSIS is grounded in the actual content of the RESUME. 
(2) Ensure the ANALYSIS does not contain "hallucinated" information that isn't present in the RESUME.
(3) Check if the skills, experiences, and qualifications mentioned in the ANALYSIS actually appear in the RESUME.

Grounded:
A grounded value of True means that the analysis meets all of the criteria.
A grounded value of False means that the analysis does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""

retrieval_relevance_instructions = """You are a teacher grading a document retrieval system. 

You will be given a JOB DESCRIPTION and a RESUME that was retrieved for that job. 

Here is the grade criteria to follow:
(1) Your goal is to identify if the RESUME is completely unrelated to the JOB DESCRIPTION
(2) If the RESUME contains ANY skills, qualifications, or experiences related to the job, consider it relevant
(3) It is OK if the RESUME has SOME information that is unrelated to the job as long as criterion (2) is met

Relevance:
A relevance value of True means that the RESUME contains relevant qualifications or experiences for the JOB DESCRIPTION.
A relevance value of False means that the RESUME is completely unrelated to the JOB DESCRIPTION.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""

# Function to process user input with automatic evaluation
def user_input(resume_text, job_description):
    chain, parser = get_conversational_chain()
    try:
        # Create Document object
        input_doc = Document(page_content=resume_text)
        
        # Call the chain with input_documents
        response = chain(
            {
                "input_documents": [input_doc],
                "context": resume_text,
                "job": job_description,
                "question": "Provide proper analysis of the resume."
            },
            return_only_outputs=True
        )
        
        # Parse the response
        parsed_response = parser.parse(response["output_text"])
        analysis_results = parsed_response.dict()
        
        # Automatic evaluation
        analysis_string = "\n".join([f"{k}: {v}" for k, v in analysis_results.items()])
        
        # Run evaluations
        evaluations = run_evaluations(resume_text, job_description, analysis_string)
        
        # Combine results
        final_result = {
            "analysis": analysis_results,
            "evaluation": evaluations
        }
        
        return final_result
    except Exception as e:
        return {"error": f"Failed to process the response: {str(e)}"}

def run_evaluations(resume_text, job_description, analysis_string):
    """Run all evaluations and return results"""
    try:
        # Evaluate relevance
        relevance_result = evaluate_relevance(resume_text, job_description, analysis_string)
        
        # Evaluate groundedness
        groundedness_result = evaluate_groundedness(resume_text, analysis_string)
        
        # Evaluate retrieval relevance
        retrieval_relevance_result = evaluate_retrieval_relevance(resume_text, job_description)
        
        # Return combined evaluation results
        return {
            "relevance": {
                "score": relevance_result["relevant"],
                "explanation": relevance_result["explanation"]
            },
            "groundedness": {
                "score": groundedness_result["grounded"],
                "explanation": groundedness_result["explanation"]
            },
            "retrieval_relevance": {
                "score": retrieval_relevance_result["relevant"],
                "explanation": retrieval_relevance_result["explanation"]
            }
        }
    except Exception as e:
        print(f"Evaluation error: {str(e)}")
        return {
            "error": f"Evaluation failed: {str(e)}"
        }

def evaluate_relevance(resume_text, job_description, analysis_result):
    """Evaluate if the analysis is relevant to the resume and job description."""
    eval_model = genai.GenerativeModel("gemini-1.5-flash-latest")
    
    prompt = f"""
{relevance_instructions}

RESUME:
{resume_text}

JOB DESCRIPTION:
{job_description}

ANALYSIS:
{analysis_result}
"""
    
    response = eval_model.generate_content(prompt)
    response_text = response.text
    
    # Parse the response for the verdict
    if "True" in response_text.lower() and "False" in response_text.lower():
        # Extract the final verdict from explanation
        lines = response_text.strip().split('\n')
        last_lines = ' '.join(lines[-3:]).lower()
        relevance = "true" in last_lines and "relevant" in last_lines
    else:
        relevance = "true" in response_text.lower() and "relevant" in response_text.lower()
    
    return {"relevant": relevance, "explanation": response_text}

def evaluate_groundedness(resume_text, analysis_result):
    """Evaluate if the analysis is grounded in the resume content."""
    eval_model = genai.GenerativeModel("gemini-1.5-flash-latest")
    
    prompt = f"""
{groundedness_instructions}

RESUME:
{resume_text}

ANALYSIS:
{analysis_result}
"""
    
    response = eval_model.generate_content(prompt)
    response_text = response.text
    
    # Parse the response for the verdict
    if "True" in response_text.lower() and "False" in response_text.lower():
        # Extract the final verdict from explanation
        lines = response_text.strip().split('\n')
        last_lines = ' '.join(lines[-3:]).lower()
        grounded = "true" in last_lines and "grounded" in last_lines
    else:
        grounded = "true" in response_text.lower() and "grounded" in response_text.lower()
    
    return {"grounded": grounded, "explanation": response_text}

def evaluate_retrieval_relevance(resume_text, job_description):
    """Evaluate if the resume is relevant to the job description."""
    eval_model = genai.GenerativeModel("gemini-1.5-flash-latest")
    
    prompt = f"""
{retrieval_relevance_instructions}

RESUME:
{resume_text}

JOB DESCRIPTION:
{job_description}
"""
    
    response = eval_model.generate_content(prompt)
    response_text = response.text
    
    # Parse the response for the verdict
    if "True" in response_text.lower() and "False" in response_text.lower():
        # Extract the final verdict from explanation
        lines = response_text.strip().split('\n')
        last_lines = ' '.join(lines[-3:]).lower()
        relevant = "true" in last_lines and "relevant" in last_lines
    else:
        relevant = "true" in response_text.lower() and "relevant" in response_text.lower()
    
    return {"relevant": relevant, "explanation": response_text}

# API Endpoint with automatic evaluation
@app.route('/api/pdf', methods=['POST'])
def pdf_qa():
    files = request.files.getlist('file')
    job_description = request.form.get('job_description')

    if not job_description:
        return jsonify({"error": "Job description is required."}), 400

    print("Processing PDF files...")
    raw_text = get_pdf_text(files)
    
    if not raw_text:
        return jsonify({"error": "No text extracted from PDF files."}), 400
        
    print("Creating text chunks and vector store...")
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks, "faiss_index_pdf")
    
    print("Analyzing resume with automatic evaluation...")
    result = user_input(raw_text, job_description)
    
    return jsonify({'response': result})

if __name__ == "__main__":
    app.run(port=5019, debug=True)