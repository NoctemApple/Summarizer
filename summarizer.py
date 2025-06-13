from transformers import BartTokenizer, BartForConditionalGeneration
import torch
import re

# Load once
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')


def split_into_sentences(text):
    sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
    return sentence_endings.split(text.strip())


def extract_key_sentences(text, num_sentences=5):
    sentences = split_into_sentences(text)
    # Naive scoring: by sentence length (you can replace with better scoring later)
    scored = sorted(sentences, key=lambda s: len(s), reverse=True)
    return scored[:num_sentences]


def summarize_text(text, max_length=130, min_length=30, num_bullets=5, mode="abstractive"):
    if len(text.split()) < 30:
        return ["⚠️ Text is too short to summarize effectively."]

    if mode == "EXTRACTIVE":
        key_sentences = extract_key_sentences(text, num_bullets)
        return [f"• {s.strip()}" for s in key_sentences]

    elif mode == "ABSTRACTIVE":
        to_summarize = text

    elif mode == "HYBRID":
        key_sentences = extract_key_sentences(text, num_sentences=7)
        to_summarize = " ".join(key_sentences)

    else:
        return [f"⚠️ Unknown mode: {mode}"]

    # Now apply BART summarization
    try:
        inputs = tokenizer.encode("summarize: " + to_summarize, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = model.generate(inputs,
                                     max_length=int(max_length),
                                     min_length=int(min_length),
                                     length_penalty=2.0,
                                     num_beams=4,
                                     early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    except Exception as e:
        return [f"⚠️ Error during summarization: {str(e)}"]

    # Bulletize
    sentences = split_into_sentences(summary)
    return [f"• {s.strip()}" for s in sentences if s.strip()][:num_bullets]
