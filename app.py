import streamlit as st

st.set_page_config(
    page_title="Financial Research Agent",
    page_icon="📈"
)

st.title("📈 Financial Research Agent")

st.write("Testing Gemini API key...")

try:
    apiKey = st.secrets["GEMINI_API_KEY"]

    st.success("Secret found successfully")

    st.write(
        f"First 10 characters: {apiKey[:10]}"
    )

    st.write(
        f"Key length: {len(apiKey)}"
    )

except Exception as e:

    st.error(
        f"Error reading secret: {e}"
    )
