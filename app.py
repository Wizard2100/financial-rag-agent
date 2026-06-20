import streamlit as st
import faiss
import pickle
import numpy as np
import pandas as pd
import json

from collections import Counter
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
# GEMINI
# =====================================

client = genai.Client(
    api_key=st.secrets["GEMINI_API_KEY"]
)

# =====================================
# LOAD MODELS
# =====================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

embedding_model = load_embedding_model()

# =====================================
# LOAD FAISS
# =====================================

@st.cache_resource
def load_faiss():
    return faiss.read_index(
        "financialIndex.faiss"
    )

index = load_faiss()

# =====================================
# LOAD CHUNKS
# =====================================

@st.cache_data
def load_chunks():
    with open(
        "companyChunks.pkl",
        "rb"
    ) as f:
        return pickle.load(f)

chunks = load_chunks()

# =====================================
# HEADER
# =====================================

st.title("📈 Financial Research Agent")

st.markdown("""
Analyze annual reports using:

- FAISS Vector Search
- Sentence Transformers
- Gemini 2.5 Flash
- Retrieval Augmented Generation (RAG)
""")

# =====================================
# KPI CARDS
# =====================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Companies",
        "3"
    )

with col2:
    st.metric(
        "Knowledge Chunks",
        len(chunks)
    )

with col3:
    st.metric(
        "Model",
        "Gemini 2.5"
    )

st.divider()

# =====================================
# INPUT
# =====================================

query = st.text_input(
    "Ask a financial question"
)

# =====================================
# MAIN PIPELINE
# =====================================

if query:

    try:

        with st.spinner(
            "Analyzing reports..."
        ):

            # ==========================
            # EMBEDDING
            # ==========================

            query_vector = embedding_model.encode(
                [query]
            )

            # ==========================
            # RETRIEVAL
            # ==========================

            D, I = index.search(
                np.array(
                    query_vector
                ).astype(
                    "float32"
                ),
                20
            )

            valid_chunks = []
            companies_used = []
            company_counter = Counter()

            for idx in I[0]:

                idx = int(idx)

                if (
                    idx >= 0
                    and
                    idx < len(chunks)
                ):

                    chunk = chunks[idx]

                    if isinstance(
                        chunk,
                        dict
                    ):

                        company = chunk.get(
                            "company",
                            "Unknown"
                        )

                        text = chunk.get(
                            "text",
                            ""
                        )

                        company_counter[
                            company
                        ] += 1

                        companies_used.append(
                            company
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
                    "No relevant chunks found."
                )

                st.stop()

            context = "\n\n".join(
                valid_chunks
            )

            # ==========================
            # PROMPT
            # ==========================

            prompt = f"""
You are a senior equity research analyst.

Analyze the retrieved annual reports.

Requirements:

1. Answer directly.
2. Explain reasoning.
3. Compare companies whenever relevant.
4. Mention financial metrics.
5. Highlight risks.
6. Highlight opportunities.
7. Use bullet points.
8. End with a conclusion.

CONTEXT:

{context}

QUESTION:

{query}
"""

            # ==========================
            # GEMINI ANSWER
            # ==========================

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )

        # =====================================
        # ANSWER
        # =====================================

        st.subheader(
            "📊 Financial Analysis"
        )

        st.success(
            response.text
        )

        # =====================================
        # COMPANY DISTRIBUTION
        # =====================================

        if len(company_counter) > 0:

            st.subheader(
                "🏢 Retrieved Company Distribution"
            )

            company_df = pd.DataFrame(
                {
                    "Company":
                        list(
                            company_counter.keys()
                        ),
                    "Chunks":
                        list(
                            company_counter.values()
                        )
                }
            )

            st.bar_chart(
                company_df.set_index(
                    "Company"
                )
            )

        # =====================================
        # COMPANIES REFERENCED
        # =====================================

        st.subheader(
            "📚 Companies Referenced"
        )

        st.write(
            " • ".join(
                sorted(
                    set(
                        companies_used
                    )
                )
            )
        )

        # =====================================
        # RETRIEVAL DETAILS
        # =====================================

        with st.expander(
            "🔍 Retrieval Details"
        ):

            scores_df = pd.DataFrame(
                {
                    "Chunk Index":
                        I[0],
                    "Distance":
                        D[0]
                }
            )

            st.dataframe(
                scores_df
            )

        # =====================================
        # RETRIEVED SOURCES
        # =====================================

        st.subheader(
            "📄 Retrieved Sources"
        )

        for i, chunk in enumerate(
            valid_chunks[:5]
        ):

            with st.expander(
                f"Source {i+1}"
            ):

                st.write(
                    chunk[:4000]
                )

    except Exception as e:

        st.error(
            f"Error: {str(e)}"
        )
