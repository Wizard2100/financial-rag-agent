import streamlit as st

st.set_page_config(
    page_title="Financial Research Agent"
)

st.title(
    "Financial Research Agent"
)

st.write(
    "Multi-Company Annual Report Analysis"
)

question = st.text_input(
    "Ask a question"
)

if st.button("Analyze"):

    st.success(
        "Backend connection coming next"
    )
