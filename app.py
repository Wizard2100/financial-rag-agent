import streamlit as st
import faiss
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer
from google import genai

# Gemini Client
client = genai.Client(
    api_key=st.secrets["GEMINI_API_KEY"]
)

# Page Settings
st.set_page_config(
    page_title="Financial Research Agent",
    page_icon="📈"
)

st.title("📈 Financial Research Agent")

# Load Embedding Model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS Index
index = faiss.read_index("financialIndex.faiss")

# Load Chunks
with open("companyChunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# Debug Info
st.write("Chunks Loaded:", len(chunks))

query = st.text_input(
    "Ask a financial question"
)

if query:

    with st.spinner("Analyzing..."):

        query_vector = model.encode([query])

        D, I = index.search(
            np.array(query_vector).astype("float32"),
            3
        )

        # Debug
        st.write("Retrieved Indices:", I)

        valid_chunks = []

        for idx in I[0]:
            if 0 <= idx < len(chunks):
                valid_chunks.append(chunks[idx])

        st.write("Retrieved Chunks:", len(valid_chunks))

        if len(valid_chunks) == 0:
            st.error(
                "No valid chunks found. Check your FAISS index and chunk file."
            )
            st.stop()

        context = "\n".join(valid_chunks)

        prompt = f"""
Context:
{context}

Question:
{query}

Answer using only the context.
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        st.write(response.text)
