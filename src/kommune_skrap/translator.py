from transformers import MarianMTModel, MarianTokenizer


def load_translation_model(model_name="Helsinki-NLP/opus-mt-no-en"):
    """Load the translation model and tokenizer."""
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return model, tokenizer


def translate_text(text, model, tokenizer):
    """Translate text from Norwegian to English."""
    translated_texts = []
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    translated_texts.append(translated_text)
    return translated_text
