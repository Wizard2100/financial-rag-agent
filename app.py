import streamlit as st
import faiss
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer
from google import genai

# -----------------------------
# Gemini Client
# -----------------------------
client = genai.Client(
    api_key=st.secrets["GEMINI_API_KEY"]
)

# -----------------------------
# Load Embedding Model
# -----------------------------
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# -----------------------------
# Load FAISS Index
# -----------------------------
index = faiss.read_index(
    "financialIndex.faiss"
)

# -----------------------------
# Load Chunks
# -----------------------------
with open(
    "companyChunks.pkl",
    "rb"
) as f:
    chunks = pickle.load(f)

# -----------------------------
# Debug
# -----------------------------
st.write(
    "Chunks Loaded:",
    len(chunks)
)

# -----------------------------
# UI
# -----------------------------
st.title(
    "📈 Financial Research Agent"
)

query = st.text_input(
    "Ask a financial question"
)

# -----------------------------
# Search
# -----------------------------
if query:

    query_vector = embedding_model.encode(
        [query]
    )

    D, I = index.search(
        np.array(query_vector).astype(
            "float32"
        ),
        3
    )

    st.write(
        "Retrieved Indices:"
    )

    st.write(I)

    valid_chunks = []

    for idx in I[0]:

        idx = int(idx)

        if idx >= 0 and idx < len(chunks):

            chunk = chunks[idx]

            valid_chunks.append(
                str(chunk)
            )

    st.write(
        "Retrieved Chunks:",
        len(valid_chunks)
    )

    if len(valid_chunks) == 0:

        st.error(
            "No valid chunks found."
        )

        st.stop()

    st.write(
        "Chunk Type:",
        type(valid_chunks[0])
    )

    st.write(
        "First Chunk Preview:"
    )

    st.write(
        valid_chunks[0][:500]
    )

    context = "\n".join(
        valid_chunks
    )

    prompt = f"""
Context:
{context}

Question:
{query}

Answer ONLY using the provided context.
If the answer is not present in the context,
say:
'I could not find that information in the reports.'
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    st.subheader(
        "Answer"
    )

    st.write(
        response.text
    )
