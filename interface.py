import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os

def setup_page():
    st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    section[data-testid="stSidebar"] { background-color: #111827; }
    p, h1, h2, h3, h4, h5 { color: #FAFAFA !important; }
    div[data-testid="stMarkdownContainer"] { color: #FAFAFA !important; }
    </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.header("📰 Project Details")
        st.markdown("**Group Name:** R&R Tech Team")
        st.divider()
        st.markdown("### 👥 Group Members")
        st.markdown("- **Virocchan A/L S.Elankoven** \n *(A24AI0092)*")
        st.markdown("- **Ryan Ang Jun Yau** \n *(A24AI0119)*")
        st.markdown("- **Muhammad Iman Hakimi Bin Abu Supian** \n *(A24AI0113)*")
        st.divider()
        st.caption("🎓 Universiti Teknologi Malaysia Kuala Lumpur")

    st.title("📰 R&R News Categorizer - Multilanguage")
    st.markdown("### What our system about?")
    st.markdown("A system that automatically categorizes news articles into topics like World, Sports, Business, and Sci/Tech.")
    st.markdown("### How to use this app:")
    st.markdown("""
    1. Type or paste a news article into the text box below.
    2. Click **Predict Category**.
    3. Explore the dashboard to see the prediction, confidence scores and training analytics.
    """)
    st.divider()


def draw_comparison_dashboard(bert_probabilities, lr_probabilities, svm_probabilities, bert_pred, lr_pred, svm_pred):
    df_bert = pd.DataFrame({"Category": list(bert_probabilities.keys()), "Confidence": list(bert_probabilities.values()), "Model": "BERT"})
    df_lr = pd.DataFrame({"Category": list(lr_probabilities.keys()), "Confidence": list(lr_probabilities.values()), "Model": "Logistic Regression"})
    df_svm = pd.DataFrame({"Category": list(svm_probabilities.keys()), "Confidence": list(svm_probabilities.values()), "Model": "Linear SVM"})
    
    df_combined = pd.concat([df_bert, df_lr, df_svm])

    st.markdown("### 📈 Confidence Breakdowns per Model")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("#### 1️⃣ BERT Confidence")
        fig1 = px.bar(df_bert, x="Confidence", y="Category", orientation="h", color="Category", text_auto=".1%", color_discrete_sequence=px.colors.qualitative.Pastel)
        fig1.update_layout(xaxis=dict(range=[0, 1], tickformat="%"), showlegend=False, height=240, paper_bgcolor="#0E1117", plot_bgcolor="#0E1117", font_color="white", margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        st.markdown("#### 2️⃣ Logistic Regression Confidence")
        fig2 = px.bar(df_lr, x="Confidence", y="Category", orientation="h", color="Category", text_auto=".1%", color_discrete_sequence=px.colors.qualitative.Safe)
        fig2.update_layout(xaxis=dict(range=[0, 1], tickformat="%"), showlegend=False, height=240, paper_bgcolor="#0E1117", plot_bgcolor="#0E1117", font_color="white", margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig2, use_container_width=True)

    with c3:
        st.markdown("#### 3️⃣ Linear SVM Confidence")
        fig3 = px.bar(df_svm, x="Confidence", y="Category", orientation="h", color="Category", text_auto=".1%", color_discrete_sequence=px.colors.qualitative.Prism)
        fig3.update_layout(xaxis=dict(range=[0, 1], tickformat="%"), showlegend=False, height=240, paper_bgcolor="#0E1117", plot_bgcolor="#0E1117", font_color="white", margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig3, use_container_width=True)

    st.write("---")
    
    st.markdown("### 📊 Cross-Model Comprehensive Comparisons")
    colA, colB = st.columns([3, 2])
    
    with colA:
        st.markdown("#### 4️⃣ Side-by-Side Category Distribution Comparison")
        fig4 = px.bar(df_combined, x="Category", y="Confidence", color="Model", barmode="group", text_auto=".1%", color_discrete_map={"BERT": "#5dfdcb", "Logistic Regression": "#fe5f55", "Linear SVM": "#3a86ff"})
        fig4.update_layout(yaxis=dict(range=[0, 1], tickformat="%"), height=350, paper_bgcolor="#0E1117", plot_bgcolor="#0E1117", font_color="white")
        st.plotly_chart(fig4, use_container_width=True)

    with colB:
        st.markdown("#### 5️⃣ Winning Confidence Gauge Metrics")
        
        bert_top_conf = bert_probabilities.get(bert_pred, 0.0) if bert_pred != "N/A" else 0.0
        lr_top_conf = lr_probabilities.get(lr_pred, 0.0) if lr_pred != "N/A" else 0.0
        svm_top_conf = svm_probabilities.get(svm_pred, 0.0) if svm_pred != "N/A" else 0.0
        
        fig5 = go.Figure()
        
        fig5.add_trace(go.Indicator(
            mode = "gauge+number", value = bert_top_conf * 100,
            domain = {'x': [0, 0.3], 'y': [0, 1]},
            title = {'text': f"BERT:<br>{bert_pred}", 'font': {'size': 11}},
            gauge = {'bar': {'color': "#5dfdcb"}, 'axis': {'range': [0, 100]}}
        ))
        
        fig5.add_trace(go.Indicator(
            mode = "gauge+number", value = lr_top_conf * 100,
            domain = {'x': [0.35, 0.65], 'y': [0, 1]},
            title = {'text': f"LR:<br>{lr_pred}", 'font': {'size': 11}},
            gauge = {'bar': {'color': "#fe5f55"}, 'axis': {'range': [0, 100]}}
        ))

        fig5.add_trace(go.Indicator(
            mode = "gauge+number", value = svm_top_conf * 100,
            domain = {'x': [0.7, 1], 'y': [0, 1]},
            title = {'text': f"SVM:<br>{svm_pred}", 'font': {'size': 11}},
            gauge = {'bar': {'color': "#3a86ff"}, 'axis': {'range': [0, 100]}}
        ))
        
        fig5.update_layout(height=350, paper_bgcolor="#0E1117", font_color="white", margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig5, use_container_width=True)


def show_image_dashboard(image_dir):
    col1, col2 = st.columns(2)

    def load_img(column, filename, title):
        column.markdown(f"#### {title}")
        path = os.path.join(image_dir, filename)
        if os.path.exists(path):
            img = Image.open(path).resize((600, 400))
            column.image(img, use_container_width=True)
        else:
            column.warning(f"Image {filename} not found.")

    with col1:
        load_img(col1, "distribution.png", "1. Dataset Distribution")
        st.write("")
        load_img(col1, "wordcloud.png", "3. Global Word Cloud")

    with col2:
        load_img(col2, "loss_curve.png", "2. Training Loss Curve")
        st.write("")
        load_img(col2, "confusion_matrix.png", "4. Confusion Matrix - BERT")