import streamlit as st
import faiss
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer
from google import genai

# =====================================
# Gemini Client
# =====================================

client = genai.Client(
    api_key=st.secrets["GEMINI_API_KEY"]
)

# =====================================
# Load Embedding Model
# =====================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# =====================================
# Load FAISS Index
# =====================================

index = faiss.read_index(
    "financialIndex.faiss"
)

# =====================================
# Load Chunks
# =====================================

with open(
    "companyChunks.pkl",
    "rb"
) as f:

    chunks = pickle.load(f)

# =====================================
# Streamlit UI
# =====================================

st.set_page_config(
    page_title="Financial Research Agent",
    page_icon="📈"
)

st.title(
    "📈 Financial Research Agent"
)

st.write(
    f"Chunks Loaded: {len(chunks)}"
)

query = st.text_input(
    "Ask a financial question"
)

# =====================================
# Search + RAG
# =====================================

if query:

    try:

        query_vector = embedding_model.encode(
            [query]
        )

        D, I = index.search(
            np.array(query_vector).astype(
                "float32"
            ),
            3
        )

        st.subheader(
            "Retrieved Indices"
        )

        st.write(I)

        valid_chunks = []

        for idx in I[0]:

            idx = int(idx)

            if 0 <= idx < len(chunks):

                valid_chunks.append(
                    str(chunks[idx])
                )

        st.subheader(
            "Retrieved Chunks Count"
        )

        st.write(
            len(valid_chunks)
        )

        if len(valid_chunks) == 0:

            st.error(
                "No chunks retrieved."
            )

            st.stop()

        # =====================================
        # DEBUG OUTPUT
        # =====================================

        st.subheader(
            "Retrieved Context"
        )

        for i, chunk in enumerate(valid_chunks):

            st.markdown(
                f"### Chunk {i+1}"
            )

            st.text(
                chunk[:1500]
            )

        context = "\n".join(
            valid_chunks
        )

        prompt = f"""
You are a financial analyst.

Use ONLY the context below.

CONTEXT:
{context}

QUESTION:
{query}

If information is not present,
say:
"I could not find that information in the reports."
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

    except Exception as e:

        st.error(
            f"ERROR: {e}"
        )
