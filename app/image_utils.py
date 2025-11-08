# image_utils.py
import numpy as np
import cv2
import json
import tensorflow
from tensorflow import keras
from keras.models import load_model
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_MODEL_PATH = os.path.join(BASE_DIR, "models", "image_emotion_model.h5")

# image model classes (in training order)
IMAGE_CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# text emotion labels we will map into
TEXT_EMOTIONS = ["anger", "fear", "joy", "love", "sadness", "surprise"]

def load_image_model():
    model = load_model(IMAGE_MODEL_PATH)
    return model

def preprocess_uploaded_image(uploaded_file, target_size=(48,48)):
    """
    uploaded_file: an io.BytesIO-like file object from streamlit.file_uploader
    returns: (orig_bgr_image, model_ready_array) where model_ready_array shape is (1,48,48,1)
    """
    # read bytes to np array
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # BGR
    if img is None:
        raise ValueError("Unable to decode uploaded image.")
    orig = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, target_size)
    arr = resized.astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=(0, -1))  # shape (1, H, W, 1)
    return orig, arr

def map_image_preds_to_text(preds):
    """
    preds: list/np array length=7 for IMAGE_CLASSES order
    returns: dict mapping TEXT_EMOTIONS to normalized probabilities
    mapping strategy:
      - 'Angry' + 'Disgust' -> 'anger'
      - 'Fear' -> 'fear'
      - 'Happy' -> 'joy'
      - 'Neutral' -> ignored (we distribute to others proportionally or leave as zero)
      - 'Sad' -> 'sadness'
      - 'Surprise' -> 'surprise'
      - 'love' has no image equiv -> set 0
    We'll add 'Neutral' into nothing (set to 0), then renormalize.
    """
    p = np.array(preds, dtype=float).flatten()
    # index map
    idx = {name: i for i, name in enumerate(IMAGE_CLASSES)}
    anger = float(p[idx['Angry']] + p[idx['Disgust']])
    fear = float(p[idx['Fear']])
    joy = float(p[idx['Happy']])
    sadness = float(p[idx['Sad']])
    surprise = float(p[idx['Surprise']])
    love = 0.0

    # create vector and normalize
    vec = np.array([anger, fear, joy, love, sadness, surprise], dtype=float)
    s = vec.sum()
    if s <= 0:
        # fallback to uniform
        vec = np.ones_like(vec) / len(vec)
    else:
        vec = vec / s
    return {emo: float(prob) for emo, prob in zip(TEXT_EMOTIONS, vec)}

def predict_image_emotion(uploaded_file, model):
    """
    uploaded_file: file-like (streamlit uploaded file)
    model: loaded keras model
    returns: dict mapping TEXT_EMOTIONS -> probabilities (normalized)
    """
    orig, arr = preprocess_uploaded_image(uploaded_file)
    preds = model.predict(arr)[0]  # shape (7,)
    mapped = map_image_preds_to_text(preds)
    return orig, mapped
