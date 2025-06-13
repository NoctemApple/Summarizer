# app.py
import streamlit as st
from summarizer import summarize_text

st.set_page_config(page_title="Text Summarizer", layout="centered")
st.title("🧠 Bullet Point Text Summarizer")

st.markdown("Paste an article or paragraph to summarize it in bullet points.")

text_input = st.text_area("Enter your text here:", height=300)
mode = st.selectbox("Choose summarization mode:", ["ABSTRACTIVE", "EXTRACTIVE", "HYBRID"])

if st.button("Summarize"):
    if text_input.strip():
        with st.spinner("Summarizing..."):
            bullets = summarize_text(
            text_input,
            max_length=130,
            min_length=30,
            num_bullets=5,
            mode=mode
        )
        st.success("✅ Summary generated!")
        st.subheader("🔍 Bullet Points:")
        for bullet in bullets:
            st.markdown(bullet)
    else:
        st.warning("Please enter some text.")
