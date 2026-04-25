from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from backend.faiss_engine import search, add_documents
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

    # Use FAISS search
    answer, score = search(query)

    return {
        "query": query,
        "answer": answer,
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
    return {"message": "Backend running 🚀 (RAG mode)"}