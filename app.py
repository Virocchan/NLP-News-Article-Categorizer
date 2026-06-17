import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json
import os
import interface as ui

st.set_page_config(
    page_title="R&R News Categorizer - Multilanguage",
    layout="wide"
)

# ==========================
# Styling
# ==========================
st.markdown("""
<style>
.stApp {
    background-color: #0E1117;
    color: #FAFAFA;
}

section[data-testid="stSidebar"] {
    background-color: #111827;
}

textarea, input {
    background-color: #1E1E1E !important;
    color: white !important;
}

p, h1, h2, h3, h4, h5 {
    color: #FAFAFA !important;
}
</style>
""", unsafe_allow_html=True)

DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(DIR, "images")

HF_MODEL = "Kimii2Dev/news-categorizer"


# ==========================
# Load Model
# ==========================
@st.cache_resource
def load_model():
    try:
        # Try Hugging Face first
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL)

    except Exception as e:
        st.warning(
            "⚠️ Could not load model from Hugging Face.\n"
            "Trying local files instead..."
        )

        st.error(f"Actual error:\n{e}")

        # Fallback to local files
        tokenizer = AutoTokenizer.from_pretrained(DIR)
        model = AutoModelForSequenceClassification.from_pretrained(DIR)

    # Load labels
    with open(os.path.join(DIR, "label_mapping.json"), "r") as f:
        labels = json.load(f)

    return tokenizer, model, labels


# ==========================
# Prediction
# ==========================
def predict(text):
    tokenizer, model, labels = load_model()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = F.softmax(outputs.logits, dim=1).squeeze()

    pred_id = torch.argmax(outputs.logits, dim=1).item()
    pred_label = labels[str(pred_id)]

    prob_dict = {
        labels[str(i)]: float(probabilities[i])
        for i in range(len(probabilities))
    }

    return pred_label, prob_dict


# ==========================
# UI
# ==========================
ui.setup_page()

tab1, tab2 = st.tabs([
    "🔍 Predict Classifier",
    "📊 Metrics Dashboard"
])

with tab1:
    st.markdown("#### Enter content for analysis:")

    text_input = st.text_area(
        "Enter text in any language (e.g. Malay, English, Chinese, Tamil and etc.) :",
        height=150
    )

    if st.button("Predict Category", type="primary"):

        if text_input.strip():

            with st.spinner("Analyzing..."):
                try:
                    label, probabilities = predict(text_input)

                    st.success(
                        f"### Predicted Category: **{label}**"
                    )

                    st.write("### Model Confidence")
                    ui.draw_confidence_chart(probabilities)

                except Exception as e:
                    st.error("Prediction failed.")
                    st.exception(e)

        else:
            st.warning("Please enter text first.")


with tab2:
    st.subheader("Training Analytics & Insights")
    ui.show_image_dashboard(IMG_DIR)