# Backend for AI Copilot RAG System
import os
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from backend.faiss_engine import search, add_documents
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    # Using the latest stable flash model available in this environment
    llm_model = genai.GenerativeModel('gemini-flash-latest')
else:
    llm_model = None

app = FastAPI()

# Request schema
class QueryRequest(BaseModel):
    query: str

# Query endpoint
@app.post("/query")
def query_docs(request: QueryRequest):
    query = request.query.strip()

    # Handle empty input
    if not query:
        return {
            "query": query,
            "answer": "Please enter a valid query.",
            "score": 0.0
        }

    # Use FAISS search to get the most relevant chunks
    matched_chunks, score = search(query, k=3)

    if not matched_chunks:
        return {
            "query": query,
            "answer": "No documents uploaded yet or no relevant matches found.",
            "score": score
        }

    context_text = "\n\n".join(matched_chunks)

    # Synthesize answer using LLM
    if llm_model:
        prompt = f"""
You are a concise AI Copilot. Your task is to provide a highly structured, short, and professional answer (maximum 5-6 lines).

### Context from Documents:
{context_text}

### User Query:
{query}

### Instructions:
1. **Direct Definition:** Start with a one-sentence clear definition.
2. **Concise Explanation:** Provide a brief explanation based on the context.
3. **Example/Formula:** Include exactly one simple example or one clearly formatted formula if present.
4. **Constraints:** Total response MUST be under 6 lines. Do NOT copy the context verbatim. Use bold for key terms.

### Answer:
"""
        try:
            response = llm_model.generate_content(prompt)
            final_answer = response.text
        except Exception as e:
            final_answer = f"Error generating answer with LLM: {str(e)}\n\n**Raw Context:**\n{context_text}"
    else:
        final_answer = f"⚠️ **LLM not configured.** Please set your `GEMINI_API_KEY` in the `.env` file.\n\n**Top Context Found:**\n{context_text}"

    return {
        "query": query,
        "answer": final_answer,
        "score": score
    }

# Upload endpoint for RAG
@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    # Extract text from PDF
    reader = PyPDF2.PdfReader(file.file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
            
    if not text.strip():
        return {"message": "No text found in PDF"}
        
    # Chunk text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    
    # Add to FAISS index
    add_documents(chunks)
    
    return {"message": f"Successfully processed '{file.filename}' into {len(chunks)} chunks"}

# Health check
@app.get("/")
def home():
    return {"message": "Backend running 🚀 (RAG mode + LLM Synthesis)"}