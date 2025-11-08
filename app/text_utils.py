# text_utils.py
import json
import pickle
import numpy as np
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences
import nltk
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ensure NLTK resources (call once in environment)
try:
    stopwords.words("english")
except:
    nltk.download("stopwords")
    nltk.download("wordnet")

STOP_WORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# config: change paths if needed
TEXT_MODEL_PATH = os.path.join(BASE_DIR, "models", "emotion_lstm_model.h5")
TOKENIZER_PATH = os.path.join(BASE_DIR, "models", "tokenizer.pkl")
MAX_LEN = 229
TEXT_EMOTIONS = ["anger", "fear", "joy", "love", "sadness", "surprise"]

# loaders
def load_text_model_and_tokenizer():
    """Load text model and tokenizer. Cached by caller (app should cache)."""
    model = keras.models.load_model(TEXT_MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

# basic cleaning pipeline (must match training)
def Removing_urls(text):
    import re
    url_pattern = re.compile(
        r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+'
        r'\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+'
        r'\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,})'
    )
    return url_pattern.sub(r'', text)

def Removing_punctuations(text):
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = text.replace('؛',"")
    text = re.sub(r'\s+', ' ', text)
    text = " ".join(text.split())
    return text.strip()

def remove_stop_words(text):
    words = [w for w in str(text).split() if w not in STOP_WORDS]
    return " ".join(words)

def lemmatization(text):
    toks = text.split()
    toks = [lemmatizer.lemmatize(t) for t in toks]
    return " ".join(toks)

def lower_case(text):
    toks = text.split()
    toks = [t.lower() for t in toks]
    return " ".join(toks)

def normalized_sentence(sentence):
    sentence = Removing_urls(sentence)
    sentence = lower_case(sentence)
    sentence = remove_stop_words(sentence)
    sentence = re.sub(r'\d+', '', sentence)
    sentence = Removing_punctuations(sentence)
    sentence = lemmatization(sentence)
    return sentence

def preprocess_for_model(text, tokenizer):
    cleaned = normalized_sentence(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, truncating='pre')
    return padded

def predict_text_emotion(text, model, tokenizer):
    """
    Returns a dict mapping TEXT_EMOTIONS to probabilities (sums to 1.0)
    """
    padded = preprocess_for_model(text, tokenizer)
    preds = model.predict(padded)[0]
    # Ensure floats (python types)
    return {emo: float(prob) for emo, prob in zip(TEXT_EMOTIONS, preds)}
