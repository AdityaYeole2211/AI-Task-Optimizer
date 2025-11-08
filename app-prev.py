import streamlit as st
import numpy as np
import pickle
import nltk 
from nltk.stem import  WordNetLemmatizer
from nltk.corpus import stopwords
import re

import tensorflow as tf
from tensorflow import keras 
from keras.preprocessing.sequence import pad_sequences

###--genai libs----###
from openai import OpenAI
from dotenv import load_dotenv
import os, json
from system_prompt import SYSTEM_PROMPT



#------------------LOAD MODEL-------------------------###
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("emotion_lstm_model.h5")   # or .keras

@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
tokenizer = load_tokenizer()
stop_words = set(stopwords.words('english'))

#gen ai model init
load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

client = OpenAI(
    api_key=GEMINI_API_KEY,
    base_url='https://generativelanguage.googleapis.com/v1beta/openai/'
)

#-------------------------------------------------------------###

##variables
emotion_labels = ["anger", "fear", "joy", "love", "sadness", "surprise"]
MAX_LEN = 229 

#####---------------TOOLS FOR GENAI MODEL-----------------------####
# TOOL: Fetch tasks from JSON 
def get_tasks(_=None):
    with open("tasks.json", "r", encoding="utf-8") as f:
        return json.load(f)

def get_emotion_preds():
    return sorted_pred_dict

available_tools = {
    "get_tasks": get_tasks,
    "get_emotion_preds":get_emotion_preds
}

### HELPER FUCNTIONS####
def lemmatization(text):
    lemmatizer= WordNetLemmatizer()

    text = text.split()

    text=[lemmatizer.lemmatize(y) for y in text]

    return " " .join(text)

def remove_stop_words(text):

    Text=[i for i in str(text).split() if i not in stop_words]
    return " ".join(Text)

def Removing_numbers(text):
    text=''.join([i for i in text if not i.isdigit()])
    return text

def lower_case(text):

    text = text.split()

    text=[y.lower() for y in text]

    return " " .join(text)

def Removing_punctuations(text):
    ## Remove punctuations
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = text.replace('؛',"", )

    ## remove extra whitespace
    text = re.sub('\s+', ' ', text)
    text =  " ".join(text.split())
    return text.strip()

def Removing_urls(text):
    url_pattern = re.compile(
        r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+'
        r'\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+'
        r'\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,})'
    )
    return url_pattern.sub(r'', text)

def remove_small_sentences(df):
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan

def normalize_text(df):
    df.Text=df.Text.apply(lambda text : Removing_urls(text))
    df.Text=df.Text.apply(lambda text : lower_case(text))
    df.Text=df.Text.apply(lambda text : remove_stop_words(text))
    df.Text=df.Text.apply(lambda text : Removing_numbers(text))
    df.Text=df.Text.apply(lambda text : Removing_punctuations(text))
    df.Text=df.Text.apply(lambda text : lemmatization(text))
    return df

def normalized_sentence(sentence):
    sentence= Removing_urls(sentence)
    sentence= lower_case(sentence)
    sentence= remove_stop_words(sentence)
    sentence= Removing_numbers(sentence)
    sentence= Removing_punctuations(sentence)
    sentence= lemmatization(sentence)
    return sentence

def preprocess_input(text):
    # 1. Apply SAME preprocessing used before training
    cleaned = normalized_sentence(text)

    # 2. Convert to sequences using trained tokenizer
    seq = tokenizer.texts_to_sequences([cleaned])

    # 3. Pad to same max length used in training
    padded = pad_sequences(seq, maxlen=MAX_LEN, truncating='pre')

    return padded

#model prediction wrapper for LIME 
def predict_proba(texts):
    seq = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(seq, maxlen=MAX_LEN, truncating='pre')
    preds = model.predict(padded)
    return preds


# -----------------------
# 5️⃣ Streamlit UI
# -----------------------
st.title("🎭 Text Emotion Detection with LIME Explainability")
st.write("Enter text below to detect emotion & see which words influenced the model.")

user_input = st.text_area("Enter your text here:", height=150)

if st.button("Analyze Emotion"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        st.write("🔄 Processing...")

        # 1. Predict Emotion
        processed = preprocess_input(user_input)
        preds = model.predict(processed)[0]
        #store predictions in a sorted dictionary (high → low confidence)
        global sorted_pred_dict
        pred_dict = {label: float(prob) for label, prob in zip(emotion_labels, preds)}
        sorted_pred_dict = dict(sorted(pred_dict.items(), key=lambda x: x[1], reverse=True))
        emotion = emotion_labels[np.argmax(preds)]
        confidence = np.max(preds) * 100

        st.success(f"### ✅ Predicted Emotion: **{emotion.upper()}** ({confidence:.2f}% confidence)")

        # 2. Show Prediction Probabilities
        st.write("#### 📊 Probability Distribution:")
        for lbl, p in zip(emotion_labels, preds):
            st.write(f"- **{lbl}**: {p:.4f}")
        
        ##----- GENAI TASK OPTMISER---------######
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]  
        def run_agent(user_text, sorted_pred_dict):
            messages.append({
                    "role": "user",
                    "content": json.dumps({
                        "emotion_prediction": sorted_pred_dict
                    })
            })

            final_output = None  # to capture last "output" step

            while True:
                response = client.chat.completions.create(
                    model="gemini-2.5-flash",
                    response_format={"type": "json_object"},
                    messages=messages
                )

                reply = response.choices[0].message.content
                print("===== RAW MODEL REPLY =====")
                print(reply)
                print("===========================")
                parsed = json.loads(reply)

                # Keep conversation going
                messages.append({"role": "assistant", "content": reply})

                step = parsed.get("step")
                content = parsed.get("content")

                # === PLAN STEP ===
                if step == "plan":
                    print(f"\n🧠 PLAN: {content}")
                    continue

                # === ACTION STEP ===
                if step == "action":
                    tool_name = parsed["function"]
                    tool_input = parsed.get("input")

                    print(f"\n⚒️ ACTION: Calling tool => {tool_name}({tool_input})")

                    result = available_tools[tool_name](tool_input)
                    messages.append({
                        "role": "user",
                        "content": json.dumps({
                            "step": "observe",
                            "output": result
                        })
                    })
                    continue

                # === OBSERVE STEP ===
                if step == "observe":
                    print(f"\n🔍 OBSERVE: {parsed['output']}")
                    continue

                # === OUTPUT STEP (FINAL) ===
                if step == "output":
                    print(f"\n✅ FINAL OUTPUT (returned to Streamlit): {content}")
                    final_output = content
                    break

            return final_output
        
        result = run_agent(user_input, sorted_pred_dict)

        st.write("### 📝 Recommended Tasks from GenAI:")

        try:
            parsed_output = json.loads(result)   # Try JSON
            st.json(parsed_output)
        except:
            st.write(result)   # Fallback: plain text

