from flask import Flask, request, jsonify
from dotenv import load_dotenv
from flask_cors import CORS
import os
import json
import time
import cv2
import numpy as np
import base64
import subprocess
from typing import List, Dict, Optional
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain.chains import LLMChain

# Load environment variables
load_dotenv()

# Configure GenAI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize Flask
app = Flask(__name__)
CORS(app)

# Pydantic Models remain the same
class TechnicalAnalysis(BaseModel):
    knowledge_score: float = Field(ge=0, le=5)
    strengths: List[str]
    areas_for_improvement: List[str]
    technical_recommendations: List[str]

class CommunicationAnalysis(BaseModel):
    clarity_score: float = Field(ge=0, le=5)
    vocabulary_score: float = Field(ge=0, le=5)
    confidence_score: float = Field(ge=0, le=5)
    communication_strengths: List[str]
    communication_improvements: List[str]

class NonVerbalAnalysis(BaseModel):
    eye_contact_score: Optional[float] = Field(default=0.0, ge=0, le=5)
    body_language_score: Optional[float] = Field(default=0.0, ge=0, le=5)
    facial_expressions: Dict[str, float] = Field(default_factory=dict)
    posture_analysis: str
    non_verbal_recommendations: List[str]

class InterviewFeedback(BaseModel):
    technical_analysis: TechnicalAnalysis
    communication_analysis: CommunicationAnalysis
    non_verbal_analysis: NonVerbalAnalysis
    overall_score: float = Field(ge=0, le=100)
    key_highlights: List[str]
    improvement_areas: List[str]
    recommendations: List[str]

def convert_video_format(input_path, output_path):
    """Convert video to mp4 format using ffmpeg"""
    try:
        command = [
            'ffmpeg',
            '-i', input_path,
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-y',  # Overwrite output file if it exists
            output_path
        ]
        
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            raise ValueError(f"FFmpeg conversion failed: {stderr.decode()}")
            
        return output_path
        
    except Exception as e:
        raise ValueError(f"Error converting video: {str(e)}")

def extract_video_frames(video_path, num_frames=20):
    """Extract random frames from the video"""
    try:
        # Convert video path to mp4 first
        mp4_path = video_path.rsplit('.', 1)[0] + '.mp4'
        converted_path = convert_video_format(video_path, mp4_path)
        
        cap = cv2.VideoCapture(converted_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            raise ValueError("No frames found in video")
        
        # Generate random frame indices
        frame_indices = sorted(np.random.choice(
            total_frames, 
            min(num_frames, total_frames), 
            replace=False
        ))
        
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Encode frame to base64
                _, buffer = cv2.imencode('.jpg', frame_rgb)
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                frames.append({
                    'frame_idx': frame_idx,
                    'frame_data': frame_base64
                })
        
        cap.release()
        
        # Clean up converted video file
        if os.path.exists(converted_path):
            os.remove(converted_path)
            
        return frames
        
    except Exception as e:
        # Clean up converted video file in case of error
        if 'converted_path' in locals() and os.path.exists(converted_path):
            os.remove(converted_path)
        raise ValueError(f"Error extracting frames: {str(e)}")
def analyze_frames(frames):
    """Analyze extracted frames using Gemini"""
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    
    analysis_results = []
    for frame in frames:
        try:
            # Decode base64 frame
            frame_data = base64.b64decode(frame['frame_data'])
            
            # Analyze frame
            response = model.generate_content(["""
                Analyze this interview frame for:
                1. Eye contact
                2. Facial expressions
                3. Body language
                4. Overall professionalism
                Provide specific observations for each category.
                """,
                {'mime_type': 'image/jpeg', 'data': frame_data}
            ])
            
            analysis_results.append({
                'frame_idx': frame['frame_idx'],
                'analysis': response.text
            })
            
        except Exception as e:
            print(f"Error analyzing frame {frame['frame_idx']}: {str(e)}")
            continue
            
    return analysis_results

def save_uploaded_file(file):
    """Save the uploaded file to a temporary location"""
    upload_dir = "temp_uploads"
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    safe_filename = f"{timestamp}_{file.filename}"
    file_path = os.path.join(upload_dir, safe_filename)
    file.save(file_path)
    return file_path

def create_feedback_chain():
    parser = PydanticOutputParser(pydantic_object=InterviewFeedback)
    
    prompt = PromptTemplate(
        template="""Analyze the following interview video analysis and responses to provide comprehensive feedback.

Interview Video Analysis:
{video_analysis}

Interview Responses:
{responses}

Provide a detailed evaluation of the candidate's performance including:
1. Technical Analysis:
   - Assess technical knowledge and skills
   - Identify strengths and areas for improvement
   - Provide specific technical recommendations

2. Communication Analysis:
   - Evaluate clarity, vocabulary, and confidence
   - Analyze speaking style and effectiveness
   - Identify communication strengths and areas for improvement

3. Non-verbal Analysis:
   - Assess eye contact and body language
   - Analyze facial expressions and posture
   - Provide recommendations for non-verbal communication

4. Overall Assessment:
   - Calculate overall score (0-100)
   - Identify key highlights
   - List areas for improvement
   - Provide actionable recommendations

{format_instructions}
""",
        input_variables=["video_analysis", "responses"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0.7,
        top_p=0.9,
        top_k=40
    )
    
    return LLMChain(llm=llm, prompt=prompt, output_parser=parser)

@app.route('/api/analyze-interview', methods=['POST'])
def analyze_interview():
    try:
        video_file = request.files.get('video')
        responses = json.loads(request.form.get('responses'))['responses']
        print(video_file)
        print(responses)

        if not responses:
            return jsonify({"error": "Interview responses are required"}), 400

        # Save the uploaded video file
        video_path = save_uploaded_file(video_file)
        print(f"Video saved to: {video_path}")

        try:
            # Extract and analyze frames
            frames = extract_video_frames(video_path)
            print(f"Extracted {len(frames)} frames")
            
            video_analysis = analyze_frames(frames)
            print("Frames analyzed")
            
            # Format video analysis for the prompt
            video_analysis_text = "\n".join(
                [f"Frame {analysis['frame_idx']}: {analysis['analysis']}" 
                 for analysis in video_analysis]
            )

            # Format responses for the prompt
            responses_text = "\n".join(
                [f"Q: {response['question']}\nA: {response['response']}" 
                 for response in responses]
            )

            # Generate feedback using LLM chain
            chain = create_feedback_chain()
            response = chain.run(
                video_analysis=video_analysis_text,
                responses=responses_text
            )
            print(response.dict())

            return jsonify({
                'status': 'success',
                'feedback': response.dict()
            })

        finally:
            # Clean up temporary file
            if os.path.exists(video_path):
                os.remove(video_path)

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

if __name__ == "__main__":
    app.run(port=5021, debug=True)




