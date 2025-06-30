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
import google.generativeai as genai
from langchain.chains import LLMChain

# Load environment variables
load_dotenv()

# Configure GenAI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize Flask
app = Flask(__name__)
CORS(app)

class ResumeAnalysis(BaseModel):
    interview_questions: list[str] = Field(
        description="List of 2-3 mixed technical and behavioral interview questions based on the job description"
    )

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_text(text)

def get_vector_store(text_chunks, vector_store_path):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(vector_store_path)
    return vector_store

def create_llm_chain():
    # Initialize the parser
    parser = PydanticOutputParser(pydantic_object=ResumeAnalysis)
    
    # Create prompt template
    prompt = PromptTemplate(
        template="""Analyze the following job description and create a mix of technical and behavioral interview questions.
        The questions should thoroughly assess the candidate's fit for the role.
        
        Job Description:
        {job_description}
        
        Generate a balanced set of questions that cover both technical skills and soft skills required for the position.
        {format_instructions}
        """,
        input_variables=["job_description"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0.7,
        top_p=0.9,
        top_k=40
    )
    
    # Create chain
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        output_parser=parser
    )
    
    return chain

@app.route('/api/interview', methods=['POST'])
def generate_interview_questions():
    try:
        # Get job description from request
        job_description = request.form.get('job_description')
        
        if not job_description:
            return jsonify({"error": "Job description is required"}), 400
            
        # Create text chunks
        text_chunks = get_text_chunks(job_description)
        
        # Create vector store
        vector_store = get_vector_store(text_chunks, "faiss_index_job")
        
        # Generate questions using LLM chain
        chain = create_llm_chain()
        response = chain.run(job_description=job_description)
        
        # Convert response to dictionary and return as JSON
        return jsonify({
            'status': 'success',
            'response': response.dict()  # Convert Pydantic model to dict
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == "__main__":
    app.run(port=5020, debug=True)