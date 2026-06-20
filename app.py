import streamlit as st
import faiss
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer
from google import genai

client = genai.Client(
    api_key=st.secrets["GEMINI_API_KEY"]
)

model = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index("financialIndex.faiss")

with open("companyChunks.pkl", "rb") as f:
    chunks = pickle.load(f)

query = st.text_input("Ask a financial question")

if query:

    query_vector = model.encode([query])

    D, I = index.search(
        np.array(query_vector).astype("float32"),
        3
    )

    context = "\n".join(
        [chunks[i] for i in I[0]]
    )

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
