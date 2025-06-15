import streamlit as st
from mistral_api.mistral import ask_mistral
from modules.chunker import chunk_text
from modules.embedder import embed_chunks, embed_query
from modules.vector_store import add_to_db, query_db

st.set_page_config(page_title="AI Teacher Prototype", layout="centered")
st.title("👾 AI Teacher (Powered by Mistral API + ChromaDB)")

tab1, tab2 = st.tabs(["Upload Study Material", "Ask a Question"])

with tab1:
    notes = st.text_area("Paste your topic or study notes here:")
    if st.button("Add to Knowledge Base"):
        chunks = chunk_text(notes)
        embeddings = embed_chunks(chunks)
        add_to_db(chunks, embeddings)
        st.success("✅ Notes added to AI Teacher's memory.")

with tab2:
    query = st.text_input("Ask a question:")
    if st.button("Get Answer"):
        query_emb = embed_query(query)
        docs = query_db(query_emb)
        context = "\n".join(docs)
        prompt = f"Use the following notes to answer:\n{context}\n\nQuestion: {query}"
        answer = ask_mistral(prompt)
        st.write("### 🧠 Answer:")
        st.success(answer)
