# app.py
import streamlit as st
from summarizer import summarize_text

st.set_page_config(page_title="Text Summarizer", layout="wide")
st.title("🧠 Bullet Point Text Summarizer")

st.markdown("Paste an article or paragraph to summarize it in bullet points. Compare **Abstractive**, **Extractive**, and **Hybrid** modes below.")

text_input = st.text_area("Enter your text here:", height=300)

with st.expander("ℹ️ What do the summarization modes mean?"):
    st.markdown("""
    **🔹 Abstractive:** Generates a summary using rephrased language, like how a human would write it. Powered by a transformer model (BART).

    **🔹 Extractive:** Selects and shows key sentences directly from the original text. Useful when you want factual fidelity.

    **🔹 Hybrid:** Picks the most relevant parts of the text using TF-IDF, then summarizes them abstractively. Balances accuracy with fluency.
    """)

if st.button("Summarize"):
    if text_input.strip():
        with st.spinner("Generating summaries..."):
            modes = ["ABSTRACTIVE", "EXTRACTIVE", "HYBRID"]
            summaries = {mode: summarize_text(text_input, mode=mode) for mode in modes}

        st.success("✅ Summaries generated!")

        col1, col2, col3 = st.columns(3)
        for col, mode in zip([col1, col2, col3], modes):
            with col:
                st.subheader(mode.capitalize())
                for bullet in summaries[mode]:
                    st.markdown(bullet)
    else:
        st.warning("Please enter some text.")
