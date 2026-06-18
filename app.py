import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json
import os
import joblib
from huggingface_hub import hf_hub_download
import numpy as np
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
BERT_REPO = "Kimii2Dev/news-categorizer"

@st.cache_resource
def load_models():
    tokenizer = None
    bert_model = None
    lr_model = None
    svm_model = None
    
    with open(os.path.join(DIR, "label_mapping.json"), "r") as f:
        labels = json.load(f)

    try:
        tokenizer = AutoTokenizer.from_pretrained(BERT_REPO, use_fast=True)
        bert_model = AutoModelForSequenceClassification.from_pretrained(BERT_REPO)
    except Exception as e:
        st.sidebar.error(f"⚠️ BERT Load Failed: {str(e)[:50]}")

    try:
        lr_model_path = hf_hub_download(repo_id=BERT_REPO, filename="bow_logistic_regression.pkl")
        lr_model = joblib.load(lr_model_path)
    except Exception as e:
        st.sidebar.error(f"⚠️ LogReg Load Failed: {str(e)[:50]}")

    try:
        svm_model_path = hf_hub_download(repo_id=BERT_REPO, filename="bow_linear_svm.pkl")
        svm_model = joblib.load(svm_model_path)
    except Exception as e:
        st.sidebar.error(f"⚠️ SVM Load Failed: {str(e)[:50]}")

    return tokenizer, bert_model, lr_model, svm_model, labels


def get_scikit_probabilities(model, text, labels):
    prob_dict = {label: 0.0 for label in labels.values()}
    label_pred = "N/A"
    
    if model is not None:
        try:
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba([text])[0]
            elif hasattr(model, "decision_function"):
                scores = model.decision_function([text])[0]
                exp_scores = np.exp(scores - np.max(scores))
                probs = exp_scores / exp_scores.sum()
            else:
                probs = None

            if hasattr(model, "classes_"):
                raw_classes = [str(c) for c in model.classes_]
                if raw_classes[0].isdigit():
                    if probs is not None:
                        prob_dict = {labels[str(c)]: float(probs[i]) for i, c in enumerate(raw_classes)}
                    pred_raw = str(model.predict([text])[0])
                    label_pred = labels[pred_raw]
                else:
                    if probs is not None:
                        prob_dict = {str(c): float(probs[i]) for i, c in enumerate(raw_classes)}
                    label_pred = str(model.predict([text])[0])
            else:
                if probs is not None:
                    prob_dict = {labels[str(i)]: float(p) for i, p in enumerate(probs)}
                pred_idx = model.predict([text])[0]
                label_pred = labels[str(pred_idx)] if str(pred_idx) in labels else str(pred_idx)
                
            if probs is None and label_pred != "N/A":
                prob_dict[label_pred] = 1.0
                
        except Exception:
            pass
            
    return label_pred, prob_dict


def predict_all(text):
    tokenizer, bert_model, lr_model, svm_model, labels = load_models()
    
    bert_label = "N/A"
    bert_prob_dict = {label: 0.0 for label in labels.values()}
    if tokenizer is not None and bert_model is not None:
        try:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
            with torch.no_grad():
                outputs = bert_model(**inputs)
            bert_probs = F.softmax(outputs.logits, dim=1).squeeze().tolist()
            bert_id = torch.argmax(outputs.logits, dim=1).item()
            bert_label = labels[str(bert_id)]
            bert_prob_dict = {labels[str(i)]: float(p) for i, p in enumerate(bert_probs)}
        except Exception:
            pass

    lr_label, lr_prob_dict = get_scikit_probabilities(lr_model, text, labels)
    svm_label, svm_prob_dict = get_scikit_probabilities(svm_model, text, labels)

    return bert_label, bert_prob_dict, lr_label, lr_prob_dict, svm_label, svm_prob_dict


ui.setup_page()

tab1, tab2 = st.tabs(["🔍 Predict Classifier", "📊 Metrics Dashboard"])

with tab1:
    st.markdown("#### Enter content for analysis:")
    text_input = st.text_area("Enter text in any language (e.g. Malay, English, Chinese, Tamil etc.) :", height=150)

    if st.button("Predict Category", type="primary"):
        if text_input:
            with st.spinner("Downloading and processing inputs across your 3 Hugging Face models..."):
                bert_label, bert_probs, lr_label, lr_probs, svm_label, svm_probs = predict_all(text_input)

                col1, col2, col3 = st.columns(3)
                with col1:
                    if bert_label != "N/A":
                        st.success(f"### BERT:\n**{bert_label}**")
                    else:
                        st.error("### BERT:\n**Failed**")
                with col2:
                    if lr_label != "N/A":
                        st.info(f"### Logistic Regression:\n**{lr_label}**")
                    else:
                        st.error("### Log Reg:\n**Failed**")
                with col3:
                    if svm_label != "N/A":
                        st.warning(f"### Linear SVM:\n**{svm_label}**")
                    else:
                        st.error("### SVM:\n**Failed**")
                
                st.write("---")
                st.write("### 📊 Model Comparisons & Confidence Dashboard")
                ui.draw_comparison_dashboard(bert_probs, lr_probs, svm_probs, bert_label, lr_label, svm_label)
        else:
            st.warning("Please enter text first.")

with tab2:
    st.subheader("Training Analytics & Insights")
    ui.show_image_dashboard(IMG_DIR)