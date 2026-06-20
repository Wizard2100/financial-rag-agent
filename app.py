import streamlit as st

st.title("Debug")

st.write("Secret exists:", "GEMINI_API_KEY" in st.secrets)

if "GEMINI_API_KEY" in st.secrets:
    key = st.secrets["GEMINI_API_KEY"]
    st.write("Length:", len(key))
    st.write("Starts with:", key[:5])
