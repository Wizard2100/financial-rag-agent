import streamlit as st
import faiss
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer
from google import genai

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Financial Research Agent",
    page_icon="📈",
    layout="wide"
)

# =====================================
# GEMINI CLIENT
# =====================================

client = genai.Client(
    api_key=st.secrets["GEMINI_API_KEY"]
)

# =====================================
# LOAD EMBEDDING MODEL
# =====================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# =====================================
# LOAD FAISS INDEX
# =====================================

index = faiss.read_index(
    "financialIndex.faiss"
)

# =====================================
# LOAD CHUNKS
# =====================================

with open(
    "companyChunks.pkl",
    "rb"
) as f:

    chunks = pickle.load(f)

# =====================================
# UI
# =====================================

st.title("📈 Financial Research Agent")

st.caption(
    f"Knowledge Base: {len(chunks)} financial chunks"
)

query = st.text_input(
    "Ask a financial question"
)

# =====================================
# SEARCH + RAG
# =====================================

if query:

    try:

        # --------------------------
        # Embed query
        # --------------------------

        query_vector = embedding_model.encode(
            [query]
        )

        # --------------------------
        # Retrieve top 20 chunks
        # --------------------------

        D, I = index.search(
            np.array(query_vector).astype(
                "float32"
            ),
            20
        )

        valid_chunks = []

        for idx in I[0]:

            idx = int(idx)

            if 0 <= idx < len(chunks):

                chunk = chunks[idx]

                if isinstance(chunk, dict):

                    company = chunk.get(
                        "company",
                        "Unknown"
                    )

                    text = chunk.get(
                        "text",
                        ""
                    )

                    valid_chunks.append(
                        f"Company: {company}\n{text}"
                    )

                else:

                    valid_chunks.append(
                        str(chunk)
                    )

        if len(valid_chunks) == 0:

            st.error(
                "No relevant information found."
            )

            st.stop()

        # --------------------------
        # Build context
        # --------------------------

        context = "\n\n".join(
            valid_chunks
        )

        # --------------------------
        # Better Prompt
        # --------------------------

        prompt = f"""
You are a professional financial analyst.

Use ONLY the information provided in the context.

Instructions:
- Answer in detail.
- Mention important financial figures.
- Compare companies if requested.
- Explain the answer clearly.
- Do not invent facts.

If the answer is not available in the context,
say:

"I could not find that information in the reports."

CONTEXT:
{context}

QUESTION:
{query}
"""

        # --------------------------
        # Gemini Response
        # --------------------------

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        # --------------------------
        # Display Answer
        # --------------------------

        st.subheader("Answer")

        st.write(
            response.text
        )

    except Exception as e:

        st.error(
            f"Error: {str(e)}"
        )
