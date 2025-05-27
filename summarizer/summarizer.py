from transformers import pipeline

summarizer = pipeline("text2text-generation", model="google/flan-t5-base")

def summarize_chunk(chunk, n_bullets=5):
    prompt = f"Summarize the following text into {n_bullets} bullet points:\n\n{chunk}"
    result = summarizer(prompt, max_length=256, min_length=60, do_sample=False)
    return result[0]['generated_text']
