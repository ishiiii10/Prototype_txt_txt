import chromadb
from chromadb.config import Settings

# ✅ New Chroma client
client = chromadb.PersistentClient(path="./vector_db")  # NEW LINE

collection = client.get_or_create_collection(name="ai_teacher")

def add_to_db(chunks, embeddings):
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        collection.add(documents=[chunk], embeddings=[emb], ids=[f"chunk_{i}"])

def query_db(query_embedding, n_results=3):
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    return results['documents'][0]
