import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

def summarize_text(text, max_length=130, min_length=30):
    if len(text.split()) < 30:
        return ["Text is too short to summarize effectively."]
    
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
    
    # Tokenize into sentences more intelligently
    sentences = sent_tokenize(summary)
    bullets = [f"• {s.strip()}" for s in sentences]
    return bullets
