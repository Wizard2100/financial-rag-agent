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

st.write(
    f"Chunks Loaded: {len(chunks)}"
)

query = st.text_input(
    "Ask a financial question"
)

# =====================================
# SEARCH
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
            5
        )

       

        valid_chunks = []

        for idx in I[0]:

            idx = int(idx)

            if 0 <= idx < len(chunks):

                chunk = chunks[idx]

                # chunk is a dictionary
                if isinstance(chunk, dict):

                    text = chunk.get(
                        "text",
                        ""
                    )

                    company = chunk.get(
                        "company",
                        "Unknown"
                    )

                    clean_text = (
                        f"Company: {company}\n\n{text}"
                    )

                    valid_chunks.append(
                        clean_text
                    )

                else:

                    valid_chunks.append(
                        str(chunk)
                    )

       

        if len(valid_chunks) == 0:

            st.error(
                "No chunks found."
            )

            st.stop()

        # =====================================
        # DEBUG CONTEXT
        # =====================================

       

        # =====================================
        # CREATE CONTEXT
        # =====================================

        context = "\n\n".join(
            valid_chunks
        )

        # =====================================
        # PROMPT
        # =====================================

        prompt = f"""
You are an expert financial analyst.

Answer ONLY from the context below.

Rules:
- Use exact numbers from the context.
- Do not confuse revenue, profit, operating income, net income, or gross margin.
- If the answer is not available in the context, reply:
"I could not find that information in the reports."

Context:
{context}

Question:
{query}

Answer:
"""

        # =====================================
        # GEMINI
        # =====================================

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
            f"ERROR: {str(e)}"
        )
