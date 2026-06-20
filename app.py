import streamlit as st

st.title("Secret Test")

if "GEMINI_API_KEY" in st.secrets:
    st.success("Secret found successfully")
else:
    st.error("Secret NOT found")
