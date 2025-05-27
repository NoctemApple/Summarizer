def format_bullets(summary_text, num_bullets=5):
    # Split by period (or other punctuation) into bullet chunks
    sentences = summary_text.split('. ')
    bullets = [s.strip().rstrip('.') for s in sentences if s.strip()]
    return ['• ' + b for b in bullets[:num_bullets]]
