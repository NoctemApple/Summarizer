# summarizer.py
from transformers import BartTokenizer, BartForConditionalGeneration
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load both models once
MODELS = {
    "BART CNN": {
        "tokenizer": BartTokenizer.from_pretrained('facebook/bart-large-cnn'),
        "model": BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    },
    "BART XSUM": {
        "tokenizer": BartTokenizer.from_pretrained('facebook/bart-large-xsum'),
        "model": BartForConditionalGeneration.from_pretrained('facebook/bart-large-xsum')
    }
}

def bulletize_summary(summary, max_bullets=5):
    sentences = summary.split('. ')
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return ["⚠️ Couldn't extract any bullet points."]

    # Score sentences by uniqueness
    vectorizer = TfidfVectorizer().fit_transform(sentences)
    scores = cosine_similarity(vectorizer, vectorizer).sum(axis=1)
    ranked = np.argsort(-scores)

    return [f"• {sentences[i]}" for i in ranked[:max_bullets]]

def summarize_text(text, model_choice="BART CNN", max_length=130, min_length=30, num_bullets=5):
    if len(text.split()) < 30:
        return ["⚠️ Text is too short to summarize effectively."]

    tokenizer = MODELS[model_choice]["tokenizer"]
    model = MODELS[model_choice]["model"]

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

    return bulletize_summary(summary, num_bullets)
