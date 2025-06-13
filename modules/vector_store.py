import chromadb
from chromadb.config import Settings

client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="./data/vector_db"))
collection = client.get_or_create_collection(name="ai_teacher_docs")

def add_to_db(chunks, embeddings):
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        collection.add(documents=[chunk], embeddings=[emb], ids=[f"doc_{i}"])

def query_db(query_embedding, n_results=3):
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    return results['documents'][0]
