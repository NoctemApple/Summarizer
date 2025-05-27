from transformers import pipeline
from summarizer.chunking import chunk_text
from .formatter import format_bullets

summarizer = pipeline("summarization", model="google/flan-t5-base", tokenizer="google/flan-t5-base")

def summarize_text(text, num_bullets=5):
    chunks = chunk_text(text)
    summaries = []

    for chunk in chunks:
        result = summarizer(chunk, max_length=150, min_length=40, do_sample=False)
        summaries.append(result[0]['summary_text'])

    # Combine and reduce summaries to final points
    combined_summary = " ".join(summaries)
    final = summarizer(combined_summary, max_length=60*num_bullets, min_length=30*num_bullets, do_sample=False)

    return format_bullets(final[0]['summary_text'], num_bullets)
