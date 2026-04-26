from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# FAISS setup for cosine similarity
dimension = model.get_sentence_embedding_dimension()
# Inner product index for cosine similarity (requires normalized vectors)
index = faiss.IndexFlatIP(dimension)  
document_store = []

def add_documents(chunks: list[str]):
    global document_store
    if not chunks:
        return
    
    # Convert chunks to embeddings
    embeddings = model.encode(chunks)
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Add to FAISS index
    index.add(np.array(embeddings))
    
    # Add to document store
    document_store.extend(chunks)

def search(query: str, k: int = 1):
    if index.ntotal == 0:
        return "No documents uploaded yet.", 0.0

    # Encode and normalize query
    query_embedding = model.encode([query])
    faiss.normalize_L2(query_embedding)
    
    distances, indices = index.search(np.array(query_embedding), k=k)
    
    # Return best match and score
    best_idx = indices[0][0]
    if best_idx == -1 or best_idx >= len(document_store):
        return "No relevant answer found.", 0.0
        
    score = float(distances[0][0])
    return document_store[best_idx], score