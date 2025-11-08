# app-shap.py
import os
import re
import math
import pickle
import random
import numpy as np
import pandas as pd
import streamlit as st
import shap
import nltk
import tensorflow

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# -----------------------
# Fix environment variable for potential oneDNN noise (optional)
# -----------------------
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # uncomment if you want to silence oneDNN note

# -----------------------
# 1) Preprocessing functions (same as training) - with raw strings to avoid regex warnings
# -----------------------
try:
    from nltk.corpus import stopwords
    _ = stopwords.words('english')  # try access
except Exception:
    nltk.download('stopwords')
    from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

from nltk.stem import WordNetLemmatizer
try:
    _ = WordNetLemmatizer()
except Exception:
    nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

def lemmatization(text):
    text = text.split()
    text = [lemmatizer.lemmatize(y) for y in text]
    return " ".join(text)

def remove_stop_words(text):
    # keep lower-case handling outside; training pipeline lowered before stop removal
    return " ".join([i for i in str(text).split() if i not in stop_words])

def Removing_numbers(text):
    return ''.join([i for i in text if not i.isdigit()])

def lower_case(text):
    text = text.split()
    text = [y.lower() for y in text]
    return " ".join(text)

def Removing_punctuations(text):
    # raw strings to avoid SyntaxWarning on escapes
    text = re.sub(r'[%s]' % re.escape(r"""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = text.replace('؛', "")
    text = re.sub(r'\s+', ' ', text)
    text = " ".join(text.split())
    return text.strip()

def Removing_urls(text):
    url_pattern = re.compile(
        r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+'
        r'\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+'
        r'\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,})'
    )
    return url_pattern.sub(r'', text)

def normalized_sentence(sentence):
    sentence = Removing_urls(sentence)
    sentence = lower_case(sentence)
    sentence = remove_stop_words(sentence)
    sentence = Removing_numbers(sentence)
    sentence = Removing_punctuations(sentence)
    sentence = lemmatization(sentence)
    return sentence

# -----------------------
# 2) App constants & file paths (update if needed)
# -----------------------
MODEL_PATH_H5 = "emotion_lstm_model.h5"
TOKENIZER_PATH = "tokenizer.pkl"
TRAIN_TXT = "train.txt"   # used to build SHAP background
MAX_LEN = 229
SHAP_BACKGROUND_SIZE = 50  # number of background samples to use (kept small for speed)

# Emotion label order must match training LabelEncoder mapping
EMOTION_LABELS = ["anger", "fear", "joy", "love", "sadness", "surprise"]

# -----------------------
# 3) Load model & tokenizer (cached)
# -----------------------
@st.cache_resource
def load_model_and_tokenizer(model_path=MODEL_PATH_H5, tokenizer_path=TOKENIZER_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model = load_model(model_path)
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

try:
    model, tokenizer = load_model_and_tokenizer()
except Exception as e:
    st.error(f"Failed to load model/tokenizer: {e}")
    st.stop()

index_word = {v: k for k, v in tokenizer.word_index.items()}

# -----------------------
# 4) Preprocess + predict helpers
# -----------------------
def preprocess_for_model(text):
    cleaned = normalized_sentence(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, truncating='post')  # use post truncation (recommended)
    return padded, cleaned

def predict_proba(texts):
    # texts: list of raw strings
    seqs = tokenizer.texts_to_sequences([normalized_sentence(t) for t in texts])
    padded = pad_sequences(seqs, maxlen=MAX_LEN, truncating='post')
    preds = model.predict(padded, verbose=0)
    return preds

# -----------------------
# 5) Build SHAP background from train.txt (cached)
# -----------------------
@st.cache_resource
def build_shap_background(train_txt_path=TRAIN_TXT, sample_size=SHAP_BACKGROUND_SIZE):
    texts = []
    if os.path.exists(train_txt_path):
        # read train.txt where each line is "text;label"
        with open(train_txt_path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                # split by last semicolon to be safe
                if ";" in line:
                    text = line.rsplit(";", 1)[0]
                else:
                    text = line
                texts.append(text)
    # fallback simple phrases if file missing or too small
    fallback = [
        "i am fine",
        "today is a normal day",
        "i feel okay",
        "this is a test sentence",
        "i am working on my project",
        "it is a typical morning",
        "i went out for a walk",
        "i ate lunch",
        "i read a book",
        "i slept well"
    ]
    if len(texts) < 10:
        texts.extend(fallback)

    # shuffle and pick sample_size
    random.shuffle(texts)
    sample = texts[:sample_size]

    # preprocess -> tokenize -> pad
    seqs = [tokenizer.texts_to_sequences([normalized_sentence(t)])[0] for t in sample]
    padded = pad_sequences(seqs, maxlen=MAX_LEN, truncating='post')
    return padded, sample

bg_padded, bg_texts = build_shap_background()

# -----------------------
# 6) Build SHAP explainer (cached)
#    Use GradientExplainer for TF2.x stability and speed
# -----------------------
@st.cache_resource
def build_shap_explainer(bg):
    try:
        # GradientExplainer tends to be more robust for TF eager mode
        expl = shap.GradientExplainer(model, bg)
    except Exception:
        # fallback - try DeepExplainer if gradient fails
        expl = shap.DeepExplainer(model, bg)
    return expl

explainer = build_shap_explainer(bg_padded)

# -----------------------
# 7) SHAP helpers: mapping sequence -> words, build HTML for dark background
# -----------------------
def sequence_to_words(seq):
    # seq is a list of ints length MAX_LEN
    words = []
    for idx in seq:
        if idx == 0:
            words.append("")  # placeholder for padding
        else:
            words.append(index_word.get(idx, "UNK"))
    return words

def make_shap_html(words, shap_vals, pos_color="#90EE90", neg_color="#FA8072"):
    """
    Build HTML for token-level highlight (dark background suitable).
    shap_vals: 1D array aligned to words list (length MAX_LEN)
    """
    vals = np.array(shap_vals, dtype=float)
    max_abs = max(1e-8, np.max(np.abs(vals)))
    norms = vals / max_abs  # -1..1

    spans = []
    for w, n in zip(words, norms):
        if not w:
            continue
        opacity = min(1.0, abs(n))
        if n > 0:
            bg = f"rgba(144,238,144,{0.2 + 0.8*opacity})"
            color = "black"
        elif n < 0:
            bg = f"rgba(250,128,114,{0.2 + 0.8*opacity})"
            color = "black"
        else:
            bg = "transparent"
            color = "white"
        span = f'<span style="background:{bg}; color:{color}; padding:2px 6px; border-radius:6px; margin:0 2px; display:inline-block;">{w}</span>'
        spans.append(span)
    html = "<div style='background:transparent; color: white; font-size:16px; line-height:2.0;'>" + " ".join(spans) + "</div>"
    return html

def st_shap_force_plot(shap_values, expected_value, feature_names=None, height=300):
    """
    Render a SHAP force plot inline in Streamlit by generating an HTML widget.
    shap_values: 1D or 2D array for a single prediction (class-specific).
    expected_value: expected value (scalar)
    """
    # shap.force_plot can produce HTML via matplotlib or JavaScript.
    try:
        fp = shap.force_plot(expected_value, shap_values, feature_names=feature_names, matplotlib=False, show=False)
        # fp is a JS/HTML object - render it
        html = f"<head>{shap.getjs()}</head><body>{fp.html()}</body>"
        st.components.v1.html(html, height=height, scrolling=True)
    except Exception:
        # fallback: create simple bar chart using absolute importances
        tv = np.array(shap_values).flatten()
        idx = np.argsort(-np.abs(tv))[:10]
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3))
        labels = [feature_names[i] for i in idx]
        vals = tv[idx]
        colors = ["#90EE90" if v > 0 else "#FA8072" for v in vals]
        ax.barh(range(len(vals))[::-1], vals[::-1], color=colors[::-1])
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(labels[::-1], color="white")
        ax.set_title("Top token contributions", color="white")
        ax.set_facecolor("black")
        fig.patch.set_facecolor('black')
        st.pyplot(fig)

# -----------------------
# 8) Streamlit UI
# -----------------------
st.set_page_config(page_title="Text Emotion Detection (SHAP)", layout="centered")
st.markdown("<h1 style='color:white;'>🎭 Text Emotion Detection (SHAP)</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:lightgray;'>Enter text below. SHAP highlights tokens that push the prediction toward/away from the chosen emotion.</p>", unsafe_allow_html=True)

user_input = st.text_area("Enter text:", height=140, value="I am feeling really good today!")

if st.button("Predict & Explain"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Predicting and computing SHAP attributions..."):
            padded, cleaned = preprocess_for_model(user_input)
            preds = model.predict(padded, verbose=0)[0]
            pred_class = int(np.argmax(preds))
            pred_prob = float(np.max(preds))
            emotion = EMOTION_LABELS[pred_class] if pred_class < len(EMOTION_LABELS) else str(pred_class)

            st.markdown(f"### <span style='color:white;'>Prediction: <b style='color:lightgreen;'>{emotion.upper()}</b>   <span style='color:lightgray;'>({pred_prob:.3f})</span></span>", unsafe_allow_html=True)

            # Compute SHAP values (GradientExplainer)
            try:
                sv = explainer.shap_values(padded)
            except Exception as ex:
                st.error("Error computing SHAP values: " + str(ex))
                sv = None

            if sv is not None:
                # sv is list of arrays (one per class) or a single array depending on explainer
                # Choose the array corresponding to predicted class
                if isinstance(sv, list) and len(sv) > pred_class:
                    shap_for_class = sv[pred_class]
                else:
                    shap_for_class = sv  # may already be class-specific

                # shap_for_class shape: (1, MAX_LEN) or (1, MAX_LEN, embed_dim)
                if shap_for_class.ndim == 3:
                    # sum across embedding dims
                    shap_agg = np.sum(shap_for_class[0], axis=1)
                else:
                    shap_agg = shap_for_class[0]

                # Map tokens -> words
                seq = padded[0].tolist()
                words = sequence_to_words(seq)

                # Build and display HTML heatmap
                html = make_shap_html(words, shap_agg)
                st.markdown("#### SHAP token attribution (highlighted):", unsafe_allow_html=True)
                st.components.v1.html(html, height=200, scrolling=True)

                # Display interactive force plot (inline)
                try:
                    # feature names: the visible words (in sequence order)
                    feature_names = [w if w else f"PAD{i}" for i, w in enumerate(words)]
                    expected_value = explainer.expected_value[pred_class] if hasattr(explainer, "expected_value") else np.mean(preds)
                    # shap values for feature_names
                    # use shap_agg as values
                    st.markdown("#### SHAP force plot (interactive):", unsafe_allow_html=True)
                    st_shap_force_plot(shap_agg, expected_value, feature_names=feature_names, height=350)
                except Exception as e:
                    st.info("Force plot could not be rendered: " + str(e))

                # Top tokens bar chart (pandas + st.bar_chart)
                token_idxs = [i for i, w in enumerate(words) if w]
                token_vals = [shap_agg[i] for i in token_idxs]
                token_words = [words[i] for i in token_idxs]
                abs_pairs = sorted(zip(token_words, token_vals), key=lambda x: abs(x[1]), reverse=True)[:10]
                if abs_pairs:
                    labels, vals = zip(*abs_pairs)
                    df = pd.DataFrame({"token": labels, "value": vals})
                    st.markdown("#### Top token contributions")
                    # color-coded bar chart via Altair for dark theme
                    import altair as alt
                    df_plot = pd.DataFrame({"token": labels, "value": vals})
                    df_plot["color"] = ["#90EE90" if v > 0 else "#FA8072" for v in vals]
                    chart = alt.Chart(df_plot).mark_bar().encode(
                        x='value:Q',
                        y=alt.Y('token:N', sort='-x'),
                        color=alt.Color('color:N', scale=None)
                    ).properties(width=600, height=300)
                    st.altair_chart(chart, use_container_width=True)
            else:
                st.info("No SHAP values available for this prediction.")
