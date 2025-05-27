from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

def chunk_text(text, max_tokens=300):
    words = text.split()
    chunks, current_chunk = [], []

    for word in words:
        current_chunk.append(word)
        token_count = len(tokenizer(" ".join(current_chunk))["input_ids"])
        if token_count >= max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks
