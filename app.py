import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json
import os
import joblib  # Added for loading the .pkl model
import interface as ui

st.set_page_config(page_title="R&R News Categorizer - Multilanguage", layout="wide")

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

MODEL_NAME = "Kimii2Dev/news-categorizer"

@st.cache_resource
def load_models():
    # 1. Load BERT Components
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    bert_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    # 2. Load Label Mapping
    with open(os.path.join(DIR, "label_mapping.json"), "r") as f:
        labels = json.load(f)

    # 3. Load Linear Regression Model and its Vectorizer
    # NOTE: Ensure 'linear_regression.pkl' and 'tfidf_vectorizer.pkl' are in your project directory
    lr_model_path = os.path.join(DIR, "linear_regression.pkl")
    vectorizer_path = os.path.join(DIR, "tfidf_vectorizer.pkl") # Standard requirement for NLP pkl models
    
    lr_model = None
    vectorizer = None
    
    if os.path.exists(lr_model_path):
        lr_model = joblib.load(lr_model_path)
    else:
        st.error("linear_regression.pkl not found in the directory!")
        
    if os.path.exists(vectorizer_path):
        vectorizer = joblib.load(vectorizer_path)
    else:
        # Fallback placeholder if you bundled the vectorizer inside the pipeline model directly
        pass

    return tokenizer, bert_model, lr_model, vectorizer, labels


def predict_all(text):
    tokenizer, bert_model, lr_model, vectorizer, labels = load_models()
    
    # --- BERT PREDICTION ---
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )
    with torch.no_grad():
        outputs = bert_model(**inputs)
    
    bert_probs = F.softmax(outputs.logits, dim=1).squeeze().tolist()
    bert_id = torch.argmax(outputs.logits, dim=1).item()
    bert_label = labels[str(bert_id)]
    bert_prob_dict = {labels[str(i)]: float(p) for i, p in enumerate(bert_probs)}

    # --- LINEAR REGRESSION PREDICTION ---
    lr_label = "N/A"
    lr_prob_dict = {label: 0.0 for label in labels.values()}
    
    if lr_model is not None:
        try:
            # Process text with vectorizer if separate, otherwise pass directly if it's a pipeline
            processed_input = vectorizer.transform([text]) if vectorizer else [text]
            
            # Get prediction probabilities
            if hasattr(lr_model, "predict_proba"):
                lr_probs = lr_model.predict_proba(processed_input)[0]
                # If classes match your json order:
                lr_prob_dict = {labels[str(i)]: float(p) for i, p in enumerate(lr_probs)}
                
                # Get max label
                lr_pred_idx = lr_model.predict(processed_input)[0]
                # Handle both numerical or string outputs from sklearn
                lr_label = labels[str(lr_pred_idx)] if str(lr_pred_idx) in labels else str(lr_pred_idx)
            else:
                # Fallback if model doesn't support predict_proba
                lr_pred = lr_model.predict(processed_input)[0]
                lr_label = labels[str(lr_pred)] if str(lr_pred) in labels else str(lr_pred)
                lr_prob_dict[lr_label] = 1.0
        except Exception as e:
            st.warning(f"Linear Regression prediction failed. Technical note: {e}")

    return bert_label, bert_prob_dict, lr_label, lr_prob_dict


ui.setup_page()

tab1, tab2 = st.tabs(["🔍 Predict Classifier", "📊 Metrics Dashboard"])

with tab1:
    st.markdown("#### Enter content for analysis:")
    text_input = st.text_area("Enter text in any language (e.g. Malay, English, Chinese, Tamil and etc.) :", height=150)

    if st.button("Predict Category", type="primary"):
        if text_input:
            with st.spinner("Analyzing with both models..."):
                bert_label, bert_probs, lr_label, lr_probs = predict_all(text_input)

                # Summary Metrics Display
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"### BERT Prediction: **{bert_label}**")
                with col2:
                    st.info(f"### Linear Regression Prediction: **{lr_label}**")
                
                st.write("---")
                st.write("### 📊 Model Comparisons & Confidence Dashboard")
                
                # Generate the 3-4 requested analysis charts
                ui.draw_comparison_dashboard(bert_probs, lr_probs, bert_label, lr_label)
                
        else:
            st.warning("Please enter text first.")

with tab2:
    st.subheader("Training Analytics & Insights")
    ui.show_image_dashboard(IMG_DIR)