import streamlit as st
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import os
import numpy as np
import google.generativeai as genai
import re
import requests
import streamlit.components.v1 as components
from PIL import Image
import easyocr
import pandas as pd
from datetime import datetime

# --- 1. CONFIGURARE MEDIU ---
API_KEY = os.getenv("GOOGLE_API_KEY", "api")
genai.configure(api_key=API_KEY)
llm = genai.GenerativeModel('gemini-2.5-flash-lite')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
MODEL_PATH = "models/phishing_distilbert_multilingual"
FEEDBACK_FILE = "data/feedback/user_feedback.csv"
DEFAULT_IMG_URL = "https://hips.hearstapps.com/hmg-prod/images/britney-spears-during-31st-annual-american-music-awards-news-photo-1589912601.jpg?crop=1.00xw:0.665xh;0,0.121xh&resize=640:*"

st.set_page_config(
    page_title="HAM OR SPAM PROTOCOL",
    layout="centered",
    initial_sidebar_state="expanded"
)

if 'last_text' not in st.session_state:
    st.session_state.last_text = None
if 'last_score' not in st.session_state:
    st.session_state.last_score = None
if 'ai_msg' not in st.session_state:
    st.session_state.ai_msg = None

# --- 3. ICONIȚE SVG ---
ICONS = {
    "skull": """<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#ff0033" stroke-width="2" stroke-linecap="square"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>""",
    "shield": """<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#00ff41" stroke-width="2" stroke-linecap="square"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><polyline points="9 12 11 14 15 10"/></svg>""",
    "eye": """<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#ffff00" stroke-width="2" stroke-linecap="square"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>""",
    "chip": """<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#00ffff" stroke-width="2" stroke-linecap="square"><rect x="4" y="4" width="16" height="16" rx="2" ry="2"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/></svg>"""
}

# --- 4. TEXTE BILINGVE ---
TEXTS = {
    "ro": {
        "sidebar_title": "Setări Sistem",
        "lang_select": "Limbă / Language:",
        "tab_text": "TEXT INPUT",
        "tab_image": "IMG SCAN (OCR)",
        "input_label": ">> INTRODU FLUXUL DE DATE:",
        "upload_label": ">> UPLOAD_SOURCE_FILE:",
        "scan_btn": ">> INITIATE_SCAN_PROTOCOL <<",
        "ocr_success": ">> DATA_EXTRACTED:",
        "error_model": ">> EROARE SISTEM: MODELUL LIPSEȘTE.",
        "error_empty": ">> EROARE: NU EXISTĂ DATE.",
        "decrypting": ">> DECRYPTING SIGNALS...",
        "ai_label": ">> NET_INTELLIGENCE:",
        "feedback_title": ">> VALIDARE HUMAN-IN-THE-LOOP:",
        "feedback_ok": "[ CORRECT ]",
        "feedback_bad": "[ WRONG ]",
        "feedback_success": ">> DATA LOGGED.",
        "res_phishing": {"title": "CRITICAL THREAT DETECTED", "prob": "PROBABILITY:", "status": "STATUS: MALICIOUS"},
        "res_legit": {"title": "SYSTEM SECURE", "prob": "INTEGRITY:", "status": "STATUS: VERIFIED"},
        "res_suspect": {"title": "UNKNOWN SIGNATURE", "prob": "RISK:", "status": "STATUS: SUSPICIOUS"}
    },
    "en": {
        "sidebar_title": "System Config",
        "lang_select": "Language / Limba:",
        "tab_text": "TEXT INPUT",
        "tab_image": "IMG SCAN (OCR)",
        "input_label": ">> DATA_STREAM_INPUT:",
        "upload_label": ">> UPLOAD_SOURCE_FILE:",
        "scan_btn": ">> INITIATE_SCAN_PROTOCOL <<",
        "ocr_success": ">> DATA_EXTRACTED:",
        "error_model": ">> SYSTEM ERROR: NEURAL NET DISCONNECTED.",
        "error_empty": ">> ERROR: NULL INPUT DETECTED.",
        "decrypting": ">> DECRYPTING SIGNALS...",
        "ai_label": ">> NET_INTELLIGENCE:",
        "feedback_title": ">> HUMAN-IN-THE-LOOP VALIDATION:",
        "feedback_ok": "[ CORRECT ]",
        "feedback_bad": "[ WRONG ]",
        "feedback_success": ">> DATA LOGGED.",
        "res_phishing": {"title": "CRITICAL THREAT DETECTED", "prob": "PROBABILITY:", "status": "STATUS: MALICIOUS"},
        "res_legit": {"title": "SYSTEM SECURE", "prob": "INTEGRITY:", "status": "STATUS: VERIFIED"},
        "res_suspect": {"title": "UNKNOWN SIGNATURE", "prob": "RISK:", "status": "STATUS: SUSPICIOUS"}
    }
}

# --- 5. STILIZARE CSS ---
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@900&family=Share+Tech+Mono&display=swap" rel="stylesheet">
<style>
    /* GLOBAL */
    .stApp {
        background-color: #020202;
        background-image: linear-gradient(rgba(0, 255, 0, 0.02) 1px, transparent 1px), linear-gradient(90deg, rgba(0, 255, 0, 0.02) 1px, transparent 1px);
        background-size: 20px 20px;
        color: #00ff41;
        font-family: 'Share Tech Mono', monospace;
    }

    /* TITLU */
    h1 {
        font-family: 'Orbitron', sans-serif;
        color: #ff00ff;
        text-align: center;
        text-shadow: 2px 2px 0px #00ffff;
        font-size: 45px !important;
        margin-bottom: 20px;
        letter-spacing: 3px;
    }

    /* INPUTS */
    .stTextArea textarea {
        background-color: #0a0a0a !important;
        color: #00ff00 !important;
        border: 1px solid #00ff00 !important;
        font-family: 'Share Tech Mono', monospace;
        font-size: 18px !important;
        border-radius: 0px;
    }
    .stTextArea textarea:focus {
        box-shadow: 0 0 10px #00ff00;
        border-color: #fff !important;
    }

    /* BUTON SCAN */
    .stButton button {
        width: 100%;
        background: transparent;
        color: #00ffff;
        font-family: 'Orbitron', sans-serif;
        font-weight: 900;
        font-size: 20px;
        border: 2px solid #00ffff !important;
        border-radius: 0px;
        text-transform: uppercase;
        transition: all 0.2s;
    }
    .stButton button:hover {
        background: #00ffff;
        color: #000;
        box-shadow: 0 0 20px #00ffff;
    }

    /* REZULTATE */
    .scan-box {
        border: 2px solid;
        padding: 20px;
        margin-top: 20px;
        background: #050505;
        display: flex;
        align-items: center;
        gap: 20px;
        font-family: 'Orbitron', sans-serif;
    }
    .box-phish { border-color: #ff0033; color: #ff0033; box-shadow: 0 0 15px rgba(255,0,51,0.3); }
    .box-safe  { border-color: #00ff41; color: #00ff41; box-shadow: 0 0 15px rgba(0,255,65,0.3); }
    .box-susp  { border-color: #ffff00; color: #ffff00; box-shadow: 0 0 15px rgba(255,255,0,0.3); }

    .res-title { font-size: 22px; font-weight: 900; margin-bottom: 5px; }
    .res-data { font-size: 18px; font-family: 'Share Tech Mono'; }

    /* AI CONSOLE */
    .ai-console {
        border-left: 3px solid #00ffff;
        background: rgba(0, 255, 255, 0.05);
        padding: 15px;
        margin-top: 15px;
        color: #e0ffff;
        font-family: 'Share Tech Mono', monospace;
        display: flex;
        gap: 15px;
    }

    /* FEEDBACK TITLE */
    .feedback-title {
        margin-top: 30px;
        margin-bottom: 10px;
        text-align: center;
        color: #666;
        font-size: 14px;
        letter-spacing: 1px;
        border-top: 1px dashed #333;
        padding-top: 15px;
        font-family: 'Orbitron', sans-serif;
    }

    /* TABURI */
    .stTabs [data-baseweb="tab-list"] { gap: 5px; }
    .stTabs [data-baseweb="tab"] { background-color: #000; color: #004400; border: 1px solid #004400; border-radius: 0px; font-family: 'Orbitron', sans-serif; }
    .stTabs [aria-selected="true"] { background-color: #00ff00 !important; color: #000 !important; border: 1px solid #00ff00 !important; }

    /* IMAGINE DEFAULT */
    .britney-default { border: 2px solid #ff00ff; padding: 4px; box-shadow: 0 0 15px #ff00ff; background: #000; margin-bottom: 20px; }

</style>
""", unsafe_allow_html=True)


# --- 6. LOGICĂ BACKEND ---
@st.cache_resource
def load_resources():
    try:
        tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
        model = TFDistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
        return tokenizer, model
    except:
        return None, None


@st.cache_resource
def load_ocr():
    return easyocr.Reader(['ro', 'en'], gpu=False)


def get_track_info(track_id):
    try:
        url = f"https://itunes.apple.com/lookup?id={track_id}"
        response = requests.get(url, timeout=2)
        data = response.json()
        if data['resultCount'] > 0:
            track = data['results'][0]
            return track['artworkUrl100'].replace('100x100bb', '600x600bb'), track['previewUrl']
    except:
        pass
    return None, None


def predict_func(text, tokenizer, model):
    encodings = tokenizer([text.lower()], truncation=True, padding=True, max_length=128, return_tensors="tf")
    output = model(encodings)
    return tf.nn.sigmoid(output.logits).numpy()[0][0]


def get_ai_msg(text, score, lang):
    v = "PHISHING" if score > 0.75 else "SAFE"
    lp = "in Romanian using cool tech slang" if lang == "ro" else "in English using hacker slang"
    prompt = f"You are a cyberpunk AI 2077. Msg: '{text}'. Verdict: {v}. Reply {lp}. Short. NO EMOJIS."
    try:
        resp = llm.generate_content(prompt)
        return re.sub(r'[^\w\s.,!?;:()"\'-]', '', resp.text).strip()
    except:
        return "AI_UPLINK_FAILED"


def save_fb(text, label, correct, lang):
    final = label if correct else (1 - label)
    df = pd.DataFrame([{"text": text, "label": final, "ts": datetime.now(), "src": "user_feedback", "lang": lang}])
    df.to_csv(FEEDBACK_FILE, mode='a', header=not os.path.exists(FEEDBACK_FILE), index=False)


# --- 7. INTERFAȚA UI ---
with st.sidebar:
    st.markdown(f"### {TEXTS['en']['sidebar_title']}")
    lang = "ro" if st.radio("LANG:", ["RO", "EN"]) == "RO" else "en"
    T = TEXTS[lang]

st.markdown('<h1>IT\'S HAM OR SPAM?</h1>', unsafe_allow_html=True)
img_box = st.empty()

# --- INPUTS ---
tab1, tab2 = st.tabs([T['tab_text'], T['tab_image']])
current_input = ""
trigger = ""

with tab1:
    txt = st.text_area(T['input_label'], height=100, key="txt_area")
    if txt: current_input = txt.strip(); trigger = txt.strip().lower()

with tab2:
    up = st.file_uploader(T['upload_label'], type=['png', 'jpg', 'jpeg'])
    if up:
        img = Image.open(up)
        st.image(img, use_container_width=True)
        with st.spinner("OCR PROCESSING..."):
            try:
                res_list = load_ocr().readtext(np.array(img), detail=0)
                res = " ".join(res_list)
                if res:
                    st.markdown(f"**{T['ocr_success']}**")
                    st.code(res)
                    current_input = res.strip();
                    trigger = res.strip().lower()
            except:
                st.error("OCR FAILED.")

# --- EASTER EGG LOGIC ---
egg = False
tid = None
if trigger == "britney spears":
    egg = True; tid = "521738935"
elif trigger == "lady gaga":
    egg = True; tid = "1451767872"

if egg and tid:
    cv, au = get_track_info(tid)
    if cv and au:
        with img_box.container():
            components.html(f"""
            <style>
                body {{ margin: 0; overflow: hidden; background: transparent; font-family: monospace; }}
                #flash {{ position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: -1; animation: strobe 0.4s infinite alternate; }}
                @keyframes strobe {{ 0% {{ background: radial-gradient(circle, rgba(255,0,255,0.15) 0%, transparent 80%); }} 100% {{ background: radial-gradient(circle, rgba(0,255,255,0.15) 0%, transparent 80%); }} }}
                .wrapper {{ display: flex; justify-content: center; align-items: center; height: 450px; width: 100%; }}
                .vinyl {{ width: 320px; height: 320px; border-radius: 50%; border: 5px solid #ff00ff; box-shadow: 0 0 50px #ff00ff, 0 0 100px #00ffff; cursor: pointer; animation: spin 1.8s linear infinite, pulse-border 0.5s infinite alternate; transition: all 0.3s; position: relative; overflow: hidden; }}
                @keyframes spin {{ 100% {{ transform: rotate(360deg); }} }}
                @keyframes pulse-border {{ from {{ border-color: #ff00ff; box-shadow: 0 0 40px #ff00ff; }} to {{ border-color: #00ffff; box-shadow: 0 0 80px #00ffff; }} }}
                .art {{ width: 100%; height: 100%; object-fit: cover; }}
                .paused {{ animation-play-state: paused !important; }}
                .overlay {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.5); display: flex; justify-content: center; align-items: center; opacity: 0; transition: 0.2s; color: #fff; font-size: 30px; font-weight: 900; text-shadow: 2px 2px #ff00ff; }}
                .vinyl:hover .overlay {{ opacity: 1; }}
            </style>
            <div id="flash"></div>
            <audio id="au" autoplay> <source src="{au}" type="audio/mp4"> </audio>
            <div class="wrapper"><div class="vinyl" id="d" onclick="toggle()"><img src="{cv}" class="art"><div class="overlay" id="btn">PAUSE</div></div></div>
            <script>
                var a=document.getElementById("au"),d=document.getElementById("d"),b=document.getElementById("btn");
                function toggle() {{ a.paused?(a.play(),d.classList.remove("paused"),b.innerHTML="PAUSE"):(a.pause(),d.classList.add("paused"),b.innerHTML="PLAY") }}
                a.onended=function(){{ window.parent.location.reload(); }}
            </script>
            """, height=460)
    else:
        st.error("ITUNES API ERROR")
else:
    with img_box.container():
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2: st.markdown(f'<div class="britney-default"><img src="{DEFAULT_IMG_URL}" style="width:100%;"></div>',
                             unsafe_allow_html=True)

# --- SCAN & FEEDBACK ---
if not egg:
    tok, mod = load_resources()
    if not tok:
        st.error(T['error_model'])
    else:
        if st.button(T['scan_btn']):
            if current_input:
                score = predict_func(current_input, tok, mod)
                st.session_state.last_score = score
                st.session_state.last_text = current_input
                st.session_state.ai_msg = get_ai_msg(current_input, score, lang)
            else:
                st.warning(T['error_empty'])

        if st.session_state.last_text:
            sc = st.session_state.last_score
            pc = sc * 100

            if sc > 0.75:
                cat, css, ic = "res_phishing", "box-phish", ICONS['skull']
            elif sc < 0.25:
                cat, css, ic = "res_legit", "box-safe", ICONS['shield']
            else:
                cat, css, ic = "res_suspect", "box-susp", ICONS['eye']

            st.markdown(f"""
            <div class="scan-box {css}">
                <div style="min-width:40px">{ic}</div>
                <div>
                    <div class="res-title">{T[cat]['title']}</div>
                    <div class="res-data">{T[cat]['prob']} {pc:.2f}% | {T[cat]['status']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            with st.spinner(T['decrypting']):
                st.markdown(
                    f"""<div class="ai-console"><div style="min-width:30px">{ICONS['chip']}</div><div>{st.session_state.ai_msg}</div></div>""",
                    unsafe_allow_html=True)

            # FEEDBACK BUTTONS
            st.markdown(f"<div class='feedback-title'>{T['feedback_title']}</div>", unsafe_allow_html=True)

            col_spacer1, col_btns, col_spacer2 = st.columns([1, 2, 1])
            pred_label = 1 if sc > 0.5 else 0

            with col_btns:
                if st.button(f"{T['feedback_ok']}", key="btn_ok", use_container_width=True):
                    save_fb(st.session_state.last_text, pred_label, True, lang)
                    st.toast(T['feedback_success'], icon="💾")

                st.write("")

                if st.button(f"{T['feedback_bad']}", key="btn_bad", use_container_width=True):
                    save_fb(st.session_state.last_text, pred_label, False, lang)
                    st.toast(T['feedback_success'], icon="💾")