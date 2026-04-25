from fastapi import FastAPI
from pydantic import BaseModel
from backend.faiss_engine import search

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

# Health check
@app.get("/")
def home():
    return {"message": "Backend running 🚀 (FAISS mode)"}