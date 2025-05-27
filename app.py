import streamlit as st
from transformers import pipeline
from summarizer.chunking import chunk_text  # Optional: only if you're chunking long texts

# Load summarizer pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Default prompt
DEFAULT_PROMPT = (
    "Summarize the following Warhammer 40K-style lore into 5 clear bullet points. "
    "Each point should focus on a major plot development or turning point in the Chapter's history. "
    "Be concise but specific. Avoid vague praise or generalizations. Return only the bullet points."
)

# UI
st.title("Lore Summarizer")

# Show default prompt
with st.expander("📌 Default Prompt Used (unless overridden)"):
    st.code(DEFAULT_PROMPT)

# Optional prompt override
user_prompt = st.text_area("✍️ Custom Prompt (optional)", value="", placeholder="Leave blank to use the default prompt")

# Final prompt logic
final_prompt = user_prompt.strip() if user_prompt.strip() else DEFAULT_PROMPT

# Lore input
input_text = st.text_area("📜 Paste Lore Text Here", height=400, placeholder="Enter your lore here...")

# Summarize button
if st.button("🧠 Summarize"):
    if not input_text.strip():
        st.warning("Please enter some lore text.")
    else:
        # Optional chunking (use only if your text is long and model input limited)
        chunks = [input_text]  # Or chunk_text(input_text)
        results = []

        for chunk in chunks:
            combined_input = f"{final_prompt}\n\n{chunk}"
            summary = summarizer(combined_input, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
            results.append(summary)

        # Show output
        st.markdown("## 📌 Summary")
        for i, result in enumerate(results):
            st.markdown(f"- {result}")
