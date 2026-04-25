from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Same documents
documents = [
    "Machine learning is a field of AI.",
    "Deep learning uses neural networks.",
    "NLP deals with text data.",
    "FAISS is used for similarity search."
]

# Convert docs → embeddings
doc_embeddings = model.encode(documents)

# Create FAISS index
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))

def search(query):
    query_embedding = model.encode([query])
    
    distances, indices = index.search(np.array(query_embedding), k=1)
    
    best_idx = indices[0][0]
    score = float(distances[0][0])

    return documents[best_idx], score