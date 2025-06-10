# app.py
import streamlit as st
from summarizer import summarize_text

st.set_page_config(page_title="Text Summarizer", layout="centered")
st.title("🧠 Bullet Point Text Summarizer")

st.markdown("Paste a long article or paragraph, and get a summary in 3–5 bullet points.")

text_input = st.text_area("Enter your text here:", height=300)

model_choice = st.selectbox("Choose summarization model:", ["BART CNN", "BART XSUM"])

if st.button("Summarize"):
    if text_input.strip():
        with st.spinner("Summarizing..."):
            bullets = summarize_text(text_input, model_choice=model_choice)
        st.success("✅ Summary generated successfully!")
        st.subheader("🔍 Bullet Points:")
        for bullet in bullets:
            st.markdown(f"- {bullet}")
    else:
        st.warning("Please enter some text to summarize.")
