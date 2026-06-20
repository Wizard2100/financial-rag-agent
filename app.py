import streamlit as st
import google.generativeai as genai

st.title("Gemini Test")

try:
    apiKey = st.secrets["GEMINI_API_KEY"]

    genai.configure(
        api_key=apiKey
    )

    model = genai.GenerativeModel(
        "gemini-1.5-flash"
    )

    response = model.generate_content(
        "Say hello"
    )

    st.success("Gemini Connected Successfully!")

    st.write(response.text)

except Exception as e:

    st.error(str(e))
