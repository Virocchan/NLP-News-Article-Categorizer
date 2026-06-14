import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json
import os
import interface as ui

# ---------------------------
# MUST be FIRST Streamlit call
# ---------------------------
st.set_page_config(page_title="R&R News Categorizer", layout="wide")

# Setup paths
DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(DIR, "images")

# ===========================
# 🔥 CHANGE THIS LINE ONLY
# ===========================
MODEL_NAME = "Kimii2Dev/rr-news-categorizer"  # <-- your Hugging Face model repo

@st.cache_resource
def load_model():
    # Load from Hugging Face (works on Streamlit Cloud)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    # Load label mapping from GitHub repo (safe)
    with open(os.path.join(DIR, "label_mapping.json"), 'r') as f:
        labels = json.load(f)

    return tokenizer, model, labels


def predict(text):
    tokenizer, model, labels = load_model()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding='max_length',
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1).squeeze().tolist()

    pred_id = torch.argmax(outputs.logits, dim=1).item()
    pred_label = labels[str(pred_id)]

    prob_dict = {
        labels[str(i)]: float(p)
        for i, p in enumerate(probs)
    }

    return pred_label, prob_dict


# -------------------
# UI START (UNCHANGED)
# -------------------
ui.setup_page()

tab1, tab2 = st.tabs(["🔍 Predict Classifier", "📊 Metrics Dashboard"])

# Tab 1
with tab1:
    st.markdown("#### Enter content for analysis:")
    text_input = st.text_area(
        "Input News Article:",
        height=150,
        placeholder="e.g., Global markets rallied on Tuesday after the central bank..."
    )

    if st.button("Predict Category", type="primary"):
        if text_input:
            with st.spinner("Analyzing..."):
                label, probabilities = predict(text_input)

                st.success(f"### Predicted Category: **{label}**")
                st.write("**Model Confidence:**")
                ui.draw_confidence_chart(probabilities)
        else:
            st.warning("Please enter text first.")

# Tab 2
with tab2:
    st.subheader("Training Analytics & Insights")
    ui.show_image_dashboard(IMG_DIR)