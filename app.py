import streamlit as st
import google.generativeai as genai

st.title("Gemini Test")

try:
    genai.configure(
        api_key=st.secrets["GEMINI_API_KEY"]
    )

    model = genai.GenerativeModel("gemini-1.5-flash")

    response = model.generate_content(
        "What is AI?"
    )

    st.success("Gemini Connected!")
    st.write(response.text)

except Exception as e:
    st.error(str(e))
