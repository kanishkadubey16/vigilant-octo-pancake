# Backend for AI Copilot RAG System
import os
from typing import List, Optional
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

# Chat message schema
class ChatMessage(BaseModel):
    role: str # "user" or "assistant"
    content: str

# Request schema
class QueryRequest(BaseModel):
    query: str
    history: List[ChatMessage] = []

# Query endpoint
@app.post("/query")
def query_docs(request: QueryRequest):
    query = request.query.strip()

    # Handle empty input
    if not query:
        return {
            "query": query,
            "answer": "Please enter a valid query.",
            "score": 0.0,
            "sources": []
        }

    # Use FAISS search to get the most relevant chunks
    matched_chunks, score = search(query, k=3)

    if not matched_chunks:
        return {
            "query": query,
            "answer": "No documents uploaded yet or no relevant matches found.",
            "score": score,
            "sources": []
        }

    context_text = "\n\n".join(matched_chunks)
    
    # Format history for prompt
    history_str = ""
    for msg in request.history[-5:]: # Last 5 messages for context
        history_str += f"{msg.role.capitalize()}: {msg.content}\n"

    # Synthesize answer using LLM
    if llm_model:
        prompt = f"""
You are a professional and concise AI Copilot. Your goal is to provide a highly structured, conversational, and accurate answer based on the context and history.

### Conversation History:
{history_str}

### Context from Documents:
{context_text}

### User Query:
{query}

### Instructions:
1. **Direct Answer:** Provide a clear, direct answer to the query.
2. **Contextual Awareness:** Use the conversation history if it helps clarify the user's intent.
3. **Conciseness:** Keep the response short and structured (max 6-8 lines).
4. **Formatting:** Use markdown (bold, lists) for readability. Include one key formula or example if relevant.
5. **Constraints:** Do not copy context verbatim. If the answer isn't in the context, say so.

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
        "score": score,
        "sources": matched_chunks
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
    return {"message": "Backend running 🚀 (Advanced RAG mode)"}