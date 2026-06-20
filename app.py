import streamlit as st
import google.generativeai as genai

genai.configure(
    api_key=st.secrets["GEMINI_API_KEY"]
)

model = genai.GenerativeModel(
    "gemini-2.5-flash"
)

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

    • Compare NVIDIA and Microsoft AI strategy

    • What are NVIDIA's growth drivers?

    • Summarize Reliance FY25
    """
)

question = st.text_input(
    "Ask a financial question"
)

if st.button("Analyze"):

    if question:

        with st.spinner("Analyzing..."):

            answer = model.generate_content(
                question
            )

            st.markdown(
                answer.text
            )
