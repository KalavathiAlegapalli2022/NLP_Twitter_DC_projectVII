import os
import re
import ast
import pickle
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
import pytesseract

# =========================
# Custom Background & Style
# =========================
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #1DA1F2, #15202B);
    color: white;
}
[data-testid="stSidebar"] {
    background: #15202B;
    color: white;
}
h1, h2, h3, h4 {
    color: #fff;
}
.css-1d391kg, .css-18e3th9 {
    color: white !important;
}
button {
    background-color: #1DA1F2 !important;
    color: white !important;
    border-radius: 8px;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# =========================
# App Config
# =========================
st.set_page_config(page_title="Twitter Disaster Prediction", layout="wide")

# -------------------------
# Constants
# -------------------------
MODEL_CANDIDATES = ["/content/mlp_model.pkl", "mlp_model.pkl"]
SCALER_CANDIDATES = ["/content/scaler.pkl", "scaler.pkl"]

NUMERIC_FEATURES = [
    "text_length", "word_length", "mention_count",
    "hashtag_count", "question_count", "exclamation_count"
]
TEXT_COL_CANDIDATES = ["text", "cleaned_text"]

IMAGE_FOLDER = "./"
IMAGE_SECTIONS = {
    "Overview": ["Project _07_Pic1.png"],
    "Tweet Classification Charts": [
        "Bar_chart_1.png",
        "Disaster_tweets and Non_Disaster_tweets.png",
        "Disaster_tweets and Non_disaster_tweets_1.png",
    ],
    "Keyword Analysis": [
        "Keyword Frequency by Target Class.png",
        "Keyword Frequency by Target Class_1.png",
        "Top_10_key_word_V_shape.png",
        "Top 10 keyword_V_shape_1.png",
    ],
    "Word Clouds": ["Word_cloud.png", "Word_cloud_1.png"],
    "Word Length Distribution": [
        "Word_lenght_disaster and Non_disaster.png",
        "word_length disaster and Non disaster_1.png",
    ],
    "Model Evaluation": ["Confusion_matrix.png", "Confusion_matrix_1.png"],
}

def _first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return paths[0]

MODEL_PATH = _first_existing(MODEL_CANDIDATES)
SCALER_PATH = _first_existing(SCALER_CANDIDATES)

# =========================
# Load model & scaler
# =========================
@st.cache_resource
def load_artifacts(model_path, scaler_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

load_ok = True
try:
    model, scaler = load_artifacts(MODEL_PATH, SCALER_PATH)
except Exception as e:
    load_ok = False
    st.error(f"❌ Failed to load model/scaler: {e}")

# =========================
# Feature Engineering
# =========================
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9#@']+")

def tokenize(s: str):
    return TOKEN_PATTERN.findall(s) if isinstance(s, str) else []

def compute_features_from_text(s: str):
    toks = tokenize(s)
    return {
        "text_length": len(s),
        "word_length": len(toks),
        "mention_count": s.count("@"),
        "hashtag_count": s.count("#"),
        "question_count": s.count("?"),
        "exclamation_count": s.count("!"),
        "tokens": toks
    }

def predict_on_text(text: str):
    feats = compute_features_from_text(text)
    feat_df = pd.DataFrame([{k: feats[k] for k in NUMERIC_FEATURES}])

    # Pad if needed
    X = feat_df.values
    expected_features = scaler.n_features_in_
    if X.shape[1] < expected_features:
        X = np.hstack([X, np.zeros((X.shape[0], expected_features - X.shape[1]))])

    Xs = scaler.transform(X)
    proba = model.predict_proba(Xs)[:, 1] if hasattr(model, "predict_proba") else model.predict(Xs)
    label = "🌪️ Disaster" if proba[0] >= 0.5 else "✅ Non-Disaster"
    return label, proba[0]

# =========================
# UI
# =========================
st.markdown("<h1 style='text-align: center;'>🐦 Twitter Disaster Prediction Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Predict whether a tweet (or image) indicates a disaster</h4>", unsafe_allow_html=True)

with st.sidebar:
    page = st.radio(
        "Navigation", 
        ["🖼️ Visual Insights", "✍️ Text Prediction", "📂 CSV Batch Prediction", "📷 Image Upload & Predict", "ℹ️ About"], 
        index=1
    )
    st.caption("Model: MLP + Scaler | Twitter NLP")

# -------------------------
# Pages
# -------------------------
if page == "🖼️ Visual Insights":
    st.subheader("📊 Visual Insights from EDA")
    for section, imgs in IMAGE_SECTIONS.items():
        with st.expander(section, expanded=False):
            cols = st.columns(2)
            for i, img in enumerate(imgs):
                path = os.path.join(IMAGE_FOLDER, img)
                if os.path.exists(path):
                    cols[i % 2].image(Image.open(path), use_column_width=True, caption=img)
                else:
                    cols[i % 2].warning(f"⚠️ Image not found: {img}")

elif page == "✍️ Text Prediction":
    st.subheader("📝 Enter Tweet Text")
    if not load_ok:
        st.stop()

    text = st.text_area("Type or paste a tweet:")
    if st.button("🔍 Predict"):
        label, proba = predict_on_text(text)
        st.success(f"{label} ({proba*100:.2f}% confidence)")

elif page == "📂 CSV Batch Prediction":
    st.subheader("📂 Batch Prediction from CSV")
    file = st.file_uploader("Upload CSV file", type="csv")
    if file:
        df = pd.read_csv(file)
        results = []
        for t in df.iloc[:, 0]:  # first column assumed to be text
            label, proba = predict_on_text(str(t))
            results.append((t, label, round(proba, 4)))

        result_df = pd.DataFrame(results, columns=["Tweet", "Prediction", "Probability"])
        st.dataframe(result_df)
        st.download_button("⬇ Download Predictions", result_df.to_csv(index=False), "predictions.csv")

elif page == "📷 Image Upload & Predict":
    st.subheader("📷 Upload an Image (Tweet Screenshot)")
    uploaded_img = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
    if uploaded_img:
        img = Image.open(uploaded_img)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        extracted_text = pytesseract.image_to_string(img)
        if extracted_text.strip():
            st.write("**Extracted Text:**")
            st.info(extracted_text)

            if st.button("🔍 Predict from Image Text"):
                label, proba = predict_on_text(extracted_text)
                st.success(f"{label} ({proba*100:.2f}% confidence)")
        else:
            st.warning("⚠️ Could not extract text from image. Try a clearer image.")

elif page == "ℹ️ About":
    st.markdown("""
    ### About
    - **Twitter-themed UI**  
    - Predict if a tweet or image of a tweet indicates a disaster  
    - Visual Insights from EDA included  
    """)
