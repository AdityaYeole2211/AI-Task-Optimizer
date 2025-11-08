# --- Path fix for Streamlit import handling ---
import sys, os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# ------------------------------------------------
import streamlit as st
import json, re, os
from dotenv import load_dotenv
from openai import OpenAI
from app.text_utils import (
    load_text_model_and_tokenizer,
    predict_text_emotion,
    TEXT_EMOTIONS
)
from app.image_utils import (
    load_image_model,
    predict_image_emotion
)
from app.db import get_session
from app.models import MoodLog
from app.utils.email_alert import (
    save_mood_log,
    last_n_logs,
    should_alert_recent_negative,
    get_hr_emails,
    send_email_alert
)
from app.auth import show_login_page, show_logout_button
from app.hr_dashboard import show_hr_dashboard

# -------------------------
# Config + GenAI init
# -------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", None)
client = None
if GEMINI_API_KEY:
    client = OpenAI(api_key=GEMINI_API_KEY, base_url='https://generativelanguage.googleapis.com/v1beta/openai/')

# -------------------------
# Cache model loads
# -------------------------
@st.cache_resource
def _load_text_components():
    return load_text_model_and_tokenizer()

@st.cache_resource
def _load_image_model():
    return load_image_model()

text_model, tokenizer = _load_text_components()
image_model = _load_image_model()

# -------------------------
# Helper: extract JSON objects from possibly-multi-block reply
# -------------------------
def extract_json_objects(text):
    blocks = re.findall(r'\{[\s\S]*?\}', text)
    parsed = []
    for b in blocks:
        try:
            parsed.append(json.loads(b))
        except Exception:
            continue
    return parsed

# -------------------------
# GenAI agent runner (resilient)
# -------------------------
def run_agent_with_tools(emotion_dict, available_tools, system_prompt, client):
    if client is None:
        return "GenAI client not configured."
    messages = [{"role": "system", "content": system_prompt}]
    messages.append({"role": "user", "content": json.dumps({"emotion_prediction": emotion_dict})})
    final_output = None

    while True:
        response = client.chat.completions.create(
            model="gemini-2.5-flash",
            response_format={"type": "json_object"},
            messages=messages
        )
        raw = response.choices[0].message.content
        print("===== RAW MODEL REPLY =====")
        print(raw)
        print("===========================")

        parsed_list = extract_json_objects(raw)
        if not parsed_list:
            try:
                parsed = json.loads(raw)
                parsed_list = [parsed]
            except Exception:
                return raw

        for parsed in parsed_list:
            # append assistant message (store as JSON string)
            messages.append({"role": "assistant", "content": json.dumps(parsed)})
            step = parsed.get("step")
            if step == "plan":
                print("🧠 PLAN:", parsed.get("content"))
                continue
            if step == "action":
                func_name = parsed.get("function")
                func_input = parsed.get("input")
                print(f"⚒️ ACTION: Calling tool => {func_name}({func_input})")
                if func_name in available_tools:
                    tool_result = available_tools[func_name](func_input)
                    messages.append({"role": "user", "content": json.dumps({"step": "observe", "output": tool_result})})
                else:
                    messages.append({"role": "user", "content": json.dumps({"step": "observe", "output": f"Unknown tool {func_name}"})})
                continue
            if step == "observe":
                print("🔍 OBSERVE:", parsed.get("output"))
                continue
            if step == "output":
                final_output = parsed.get("content")
                print("✅ FINAL OUTPUT:", final_output)
                return final_output

# -------------------------
# Tools exposed to GenAI
# -------------------------
def get_tasks_tool(_=None):
    base_dir = os.path.dirname(__file__)
    tasks_path = os.path.join(base_dir, "tasks.json")
    with open(tasks_path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_emotion_preds_tool(_=None):
    return st.session_state.get("merged_emotion", {})

available_tools = {"get_tasks": get_tasks_tool, "get_emotion_preds": get_emotion_preds_tool}

# -------------------------
# Session defaults
# -------------------------
if "text_raw" not in st.session_state:
    st.session_state["text_raw"] = ""
if "uploaded_file" not in st.session_state:
    st.session_state["uploaded_file"] = None
if "text_emotion" not in st.session_state:
    st.session_state["text_emotion"] = None
if "image_emotion" not in st.session_state:
    st.session_state["image_emotion"] = None
if "merged_emotion" not in st.session_state:
    st.session_state["merged_emotion"] = None
if "genai_calls" not in st.session_state:
    st.session_state["genai_calls"] = 0  # resets on refresh as requested

# -------------------------
# UI layout
# -------------------------

# ---------------------------------------------------
# ROLE-BASED ROUTING (insert this near top of app.py)
# ---------------------------------------------------


# Show login page if not logged in
if "logged_in" not in st.session_state or not st.session_state["logged_in"]:
    show_login_page()
    st.stop()  # wait until login happens

# If HR logs in — show only HR dashboard, hide emotion detection UI
if st.session_state.get("role") == "hr":
    show_logout_button()
    show_hr_dashboard()
    st.stop()

st.set_page_config(layout="wide", page_title="Multi-Modal Emotion + Tasks")

# Logged in
st.sidebar.markdown(f"**👤 {st.session_state['name']} ({st.session_state['email']})**")
show_logout_button()

#----- MAIN UI---------#

st.title("🎭 Multi-Modal Emotion Detection (Text + Image)")
st.write("Provide **text** and **image**. Only when both are present you can combine and get a recommended task.")

col1, col2 = st.columns(2)

with col1:
    st.header("Enter text")
    text_input = st.text_area("Type your text here:", height=140, value=st.session_state["text_raw"])
    # store raw text but do not predict yet
    st.session_state["text_raw"] = text_input

with col2:
    st.header("Upload face image")
    uploaded = st.file_uploader("Upload face image (jpg/png)", type=["jpg","jpeg","png"])
    if uploaded is not None:
        # store file-like object in session but don't predict yet
        st.session_state["uploaded_file"] = uploaded

# Combine button - disabled until both inputs exist
both_present = bool(st.session_state["text_raw"].strip()) and (st.session_state["uploaded_file"] is not None)
combine_col1, combine_col2 = st.columns([3,1])
with combine_col1:
    combine_btn = st.button("Combine & Recommend Task", disabled=not both_present)

# Show results area
st.markdown("---")
results_col, right_col = st.columns([3,1])

with results_col:
    if not both_present:
        st.info("Please provide both text and image to enable recommendation.")
    elif combine_btn:
        # Run inference for both modalities now
        with st.spinner("Running multimodal predictions..."):
            # Text prediction
            try:
                text_preds = predict_text_emotion(st.session_state["text_raw"], text_model, tokenizer)
                st.session_state["text_emotion"] = text_preds
            except Exception as e:
                st.error("Text model error: " + str(e))
                text_preds = None

            # Image prediction
            try:
                # predict_image_emotion returns (orig_img, mapped_dict)
                orig_img, img_mapped = predict_image_emotion(st.session_state["uploaded_file"], image_model)
                st.session_state["image_emotion"] = img_mapped
            except Exception as e:
                st.error("Image model error: " + str(e))
                img_mapped = None

            # Build merged weighted vector: merged = 0.4 * text + 0.6 * image
            merged = {}
            for emo in TEXT_EMOTIONS:
                t = text_preds.get(emo, 0.0) if text_preds else 0.0
                i = img_mapped.get(emo, 0.0) if img_mapped else 0.0
                merged[emo] = 0.4 * t + 0.6 * i

            # Normalize merged
            s = sum(merged.values())
            if s > 0:
                for k in merged:
                    merged[k] = merged[k] / s
            else:
                for k in merged:
                    merged[k] = 1.0 / len(merged)

            st.session_state["merged_emotion"] = merged

            # ---------------------------------------------------
            # SAVE MOOD LOG + ALERT HR IF CONSECUTIVE NEGATIVE
            # ---------------------------------------------------
            from app.utils.email_alert import (
                save_mood_log,
                should_alert_recent_negative,
                get_hr_emails,
                send_email_alert,
            )

            if "user_id" in st.session_state:
                try:
                    ml = save_mood_log(st.session_state["user_id"], merged)
                    st.success("✅ Mood log saved successfully.")
                except Exception as e:
                    st.warning(f"⚠️ Could not save mood log: {e}")

                # Check last 2 moods for consecutive negative emotions
                try:
                    uid = st.session_state["user_id"]
                    if should_alert_recent_negative(uid, n=2):
                        hr_emails = get_hr_emails()
                        if hr_emails:
                            subject = f"[Alert] Consecutive Negative Emotions Detected — {st.session_state.get('username', 'Employee')}"
                            body = (
                                f"Employee: {st.session_state.get('username', 'unknown')}\n"
                                f"User ID: {uid}\n"
                                f"Recent emotion trends indicate possible distress.\n"
                                "Please review this employee’s recent logs and reach out if needed.\n\n"
                                "— Automated Mood Tracker System"
                            )
                            try:
                                send_email_alert(hr_emails, subject, body)
                                st.info("📧 HR has been notified of consecutive negative moods.")
                            except Exception as e:
                                st.warning(f"⚠️ Failed to send HR email alert: {e}")
                except Exception as e:
                    print("Alert check failed:", e)

        # Present only labels + confidences (B format)
        def top_label_and_conf(d):
            emo = max(d, key=d.get)
            conf = d[emo]
            return emo, conf

        # Text top
        if text_preds:
            t_label, t_conf = top_label_and_conf(text_preds)
            st.markdown(f"**📝 Text Emotion:** {t_label.capitalize()} ({t_conf:.2f})")
        else:
            st.markdown("**📝 Text Emotion:** —")

        # Image top
        if img_mapped:
            i_label, i_conf = top_label_and_conf(img_mapped)
            st.markdown(f"**🖼️ Image Emotion:** {i_label.capitalize()} ({i_conf:.2f})")
            # show uploaded image preview (already loaded inside predict_image_emotion as orig_img)
            try:
                st.image(orig_img[:, :, ::-1], caption="Uploaded image", use_container_width=True)
            except Exception:
                pass
        else:
            st.markdown("**🖼️ Image Emotion:** —")

        # Merged top
        m_label, m_conf = top_label_and_conf(merged)
        st.markdown(f"**🔗 Final Merged Emotion:** {m_label.capitalize()} ({m_conf:.2f})")

        # Expanders for detailed probabilities
        with st.expander("Show full probability breakdown (text / image / merged)"):
            def sort_and_round(d):
                return {k: round(v, 4) for k, v in sorted(d.items(), key=lambda x: x[1], reverse=True)}

            st.write("**Text probabilities**")
            st.json(sort_and_round(text_preds) if text_preds else {})
            st.write("**Image probabilities (mapped to text labels)**")
            st.json(sort_and_round(img_mapped) if img_mapped else {})
            st.write("**Merged probabilities**")
            st.json(sort_and_round(merged))

        # Call GenAI recommender exactly once
        if client is None:
            st.info("GenAI not configured (GEMINI_API_KEY missing). Skipping recommendation.")
        else:
            with st.spinner("Asking GenAI for recommended task..."):
                final = run_agent_with_tools(
                    merged, available_tools, __import__("system_prompt").SYSTEM_PROMPT, client
                )
                st.markdown("### 📝 Recommended Task from GenAI")
                try:
                    parsed_final = json.loads(final)
                    st.json(parsed_final)
                except Exception:
                    st.write(final)

                st.session_state["genai_calls"] += 1

with right_col:
    # place-holder or instructions
    st.markdown("#### Instructions")
    st.write("1. Type your text on the left.\n2. Upload a face image on the right.\n3. Click **Combine & Recommend Task** once both are present.")

# -------------------------
# Bottom-left small GenAI counter (minimal gray text)
# -------------------------
# CSS to position a tiny box bottom-left
counter_html = f"""
<div style="
    position: fixed;
    left: 8px;
    bottom: 8px;
    background: rgba(255,255,255,0.85);
    color: #444;
    padding: 6px 8px;
    font-size: 12px;
    border-radius: 6px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.2);
    z-index: 9999;
">
⚡ GenAI Calls: {st.session_state['genai_calls']}
</div>
"""
st.markdown(counter_html, unsafe_allow_html=True)
