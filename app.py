import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
import sys
import joblib
from huggingface_hub import hf_hub_download
import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(DIR, "images")

if DIR not in sys.path:
    sys.path.append(DIR)

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

BERT_REPO = "Kimii2Dev/news-categorizer"

@st.cache_resource
def load_models():
    tokenizer = None
    bert_model = None
    bow_lr_model = None
    pipe_lr_model = None
    
    bert_labels = {"0": "World", "1": "Sports", "2": "Business", "3": "Sci/Tech"}
    scikit_labels = {"1": "World", "2": "Sports", "3": "Business", "4": "Sci/Tech"}

    try:
        tokenizer = AutoTokenizer.from_pretrained(BERT_REPO, use_fast=True)
        bert_model = AutoModelForSequenceClassification.from_pretrained(BERT_REPO)
    except Exception as e:
        st.sidebar.error(f"⚠️ BERT Load Failed: {str(e)[:50]}")

    try:
        bow_lr_path = hf_hub_download(repo_id=BERT_REPO, filename="bow_logistic_regression.pkl")
        bow_lr_model = joblib.load(bow_lr_path)
    except Exception as e:
        st.sidebar.error(f"⚠️ BoW LogReg Load Failed: {str(e)[:50]}")

    try:
        pipe_lr_path = hf_hub_download(repo_id=BERT_REPO, filename="pipeline_logistic_regression.pkl")
        pipe_lr_model = joblib.load(pipe_lr_path)
    except Exception as e:
        st.sidebar.error(f"⚠️ Pipeline LogReg Load Failed: {str(e)[:50]}")

    return tokenizer, bert_model, bow_lr_model, pipe_lr_model, bert_labels, scikit_labels


def get_scikit_predictions(model, text, scikit_labels):
    prob_dict = {label: 0.0 for label in scikit_labels.values()}
    label_pred = "N/A"
    
    if model is not None:
        try:
            pred_raw = str(model.predict([text])[0])
            label_pred = scikit_labels.get(pred_raw, "N/A")
            
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba([text])[0]
            elif hasattr(model, "decision_function"):
                scores = model.decision_function([text])[0]
                exp_scores = np.exp(scores - np.max(scores))
                probs = exp_scores / exp_scores.sum()
            else:
                probs = None

            if probs is not None and hasattr(model, "classes_"):
                for idx, class_val in enumerate(model.classes_):
                    class_str = str(class_val)
                    cat_name = scikit_labels.get(class_str)
                    if cat_name:
                        prob_dict[cat_name] = float(probs[idx])
            elif label_pred != "N/A":
                prob_dict[label_pred] = 1.0
        except Exception:
            pass
            
    return label_pred, prob_dict


def predict_all(text):
    tokenizer, bert_model, bow_lr_model, pipe_lr_model, bert_labels, scikit_labels = load_models()
    
    bert_label = "N/A"
    bert_prob_dict = {label: 0.0 for label in bert_labels.values()}
    
    if tokenizer is not None and bert_model is not None:
        try:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
            with torch.no_grad():
                outputs = bert_model(**inputs)
            bert_probs = F.softmax(outputs.logits, dim=1).squeeze().tolist()
            bert_id = str(torch.argmax(outputs.logits, dim=1).item())
            bert_label = bert_labels.get(bert_id, "N/A")
            
            for idx, p in enumerate(bert_probs):
                cat_name = bert_labels.get(str(idx))
                if cat_name:
                    bert_prob_dict[cat_name] = float(p)
        except Exception:
            pass

    bow_lr_label, bow_lr_probs = get_scikit_predictions(bow_lr_model, text, scikit_labels)
    pipe_lr_label, pipe_lr_probs = get_scikit_predictions(pipe_lr_model, text, scikit_labels)

    return bert_label, bert_prob_dict, bow_lr_label, bow_lr_probs, pipe_lr_label, pipe_lr_probs


ui.setup_page()

tab1, tab2 = st.tabs(["🔍 Predict Classifier", "📊 Metrics Dashboard"])

with tab1:
    st.markdown("#### Enter content for analysis:")
    text_input = st.text_area("Enter text in any language (e.g. Malay, English, Chinese, Tamil etc.) :", height=150)

    if st.button("Predict Category", type="primary"):
        if text_input:
            with st.spinner("Processing inputs across models..."):
                bert_label, bert_probs, bow_label, bow_probs, pipe_label, pipe_probs = predict_all(text_input)

                col1, col2, col3 = st.columns(3)
                with col1:
                    if bert_label != "N/A":
                        st.success(f"### BERT:\n**{bert_label}**")
                    else:
                        st.error("### BERT:\n**Failed**")
                with col2:
                    if bow_label != "N/A":
                        st.info(f"### BoW LogReg:\n**{bow_label}**")
                    else:
                        st.error("### BoW LogReg:\n**Failed**")
                with col3:
                    if pipe_label != "N/A":
                        st.warning(f"### Pipeline LogReg:\n**{pipe_label}**")
                    else:
                        st.error("### Pipeline LogReg:\n**Failed**")
                
                st.write("---")
                st.write("### 📊 Model Comparisons & Confidence Dashboard")
                ui.draw_comparison_dashboard(bert_probs, bow_probs, pipe_probs, bert_label, bow_label, pipe_label)
        else:
            st.warning("Please enter text first.")

with tab2:
    st.subheader("Dataset Visualizations & Analytics")
    ui.show_image_dashboard(IMG_DIR)