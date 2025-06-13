from transformers import BartTokenizer, BartForConditionalGeneration
import torch
import re

# Load tokenizer and model once
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

def split_into_sentences(text):
    # Basic sentence splitter using regex
    sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
    return sentence_endings.split(text.strip())

def summarize_text(text, max_length=130, min_length=30, num_bullets=5, mode="abstractive"):

    max_length = int(max_length)
    min_length = int(min_length)
    num_bullets = int(num_bullets)

    if len(text.split()) < 30:
        return ["⚠️ Text is too short to summarize effectively."]

    try:
        inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = model.generate(inputs,
                                     max_length=max_length,
                                     min_length=min_length,
                                     length_penalty=2.0,
                                     num_beams=4,
                                     early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    except Exception as e:
        return [f"⚠️ Error during summarization: {str(e)}"]

    # Convert summary into bullet points using regex
    sentences = split_into_sentences(summary)
    bullets = [f"• {s.strip()}" for s in sentences if s.strip()]
    return bullets[:num_bullets]
