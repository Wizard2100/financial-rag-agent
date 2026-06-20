import streamlit as st

st.set_page_config(
    page_title="Financial Research Agent",
    page_icon="📈"
)

st.title("📈 Financial Research Agent")

st.write(
    """
    Analyze annual reports of:

    • NVIDIA
    • Microsoft
    • Reliance

    Example Questions:
    - Compare NVIDIA and Microsoft AI strategy
    - What are NVIDIA's growth drivers?
    - Summarize Reliance FY25
    """
)

question = st.text_input(
    "Ask a financial question"
)

if st.button("Analyze"):

    if question:

        st.info(
            f"Question Received: {question}"
        )

        st.success(
            "RAG backend will be connected next."
        )
