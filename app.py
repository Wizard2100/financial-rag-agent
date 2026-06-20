```python
import streamlit as st
import faiss
import pickle
import numpy as np
import pandas as pd
import re

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

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

index = faiss.read_index(
    "financialIndex.faiss"
)

with open(
    "companyChunks.pkl",
    "rb"
) as f:
    chunks = pickle.load(f)

# =====================================
# HEADER
# =====================================

st.title("📈 Financial Research Agent")
st.markdown(
    "Analyze NVIDIA, Microsoft and Reliance annual reports using Retrieval-Augmented Generation (RAG)."
)

# KPI CARDS

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Companies", "3")

with col2:
    st.metric("Knowledge Chunks", len(chunks))

with col3:
    st.metric("Model", "Gemini 2.5")

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

        with st.spinner("Analyzing reports..."):

            query_vector = embedding_model.encode(
                [query]
            )

            D, I = index.search(
                np.array(query_vector).astype(
                    "float32"
                ),
                50
            )
            st.subheader("Similarity Scores")

            scores_df = pd.DataFrame({
                "Chunk Index": I[0],
                "Similarity Score": D[0]
            })

st.dataframe(scores_df)

            valid_chunks = []
            companies_used = set()

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

                        companies_used.add(
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
                    "No relevant information found."
                )

                st.stop()

            context = "\n\n".join(
                valid_chunks
            )

            prompt = f"""
You are a senior equity research analyst.

Carefully analyze ALL retrieved context before answering.

Rules:
- Use information from multiple chunks whenever possible.
- Perform comparisons if the question asks for them.
- Extract numerical values and financial metrics.
- Explain your reasoning.
- Summarize findings in bullet points.
- Give a final conclusion.

If the answer is partially available, provide the available information instead of saying you cannot find it.

CONTEXT:
{context}

QUESTION:
{query}
"""

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )

        # =====================================
        # ANSWER
        # =====================================

        st.subheader("📊 Financial Analysis")

        st.success(
            response.text
        )

        # =====================================
        # SOURCES
        # =====================================

        st.subheader("📚 Companies Referenced")

        st.write(
            " • ".join(
                sorted(companies_used)
            )
        )

        # =====================================
        # THEMES CHART
        # =====================================

        words = re.findall(
            r"\b[a-zA-Z]{4,}\b",
            context.lower()
        )

        stopwords = {
            "company","revenue","income","would",
            "could","their","there","which",
            "these","those","about","after",
            "before","other","using","from",
            "have","been","were","with"
        }

        words = [
            w for w in words
            if w not in stopwords
        ]

        top_words = Counter(
            words
        ).most_common(10)

        if len(top_words) > 0:

            df = pd.DataFrame(
                top_words,
                columns=[
                    "Keyword",
                    "Frequency"
                ]
            )

            st.subheader(
                "📈 Key Financial Themes"
            )

            st.bar_chart(
                df.set_index(
                    "Keyword"
                )
            )

    except Exception as e:

        st.error(
            f"Error: {str(e)}"
        )
```
