import streamlit st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os


def setup_page():
    st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }

    section[data-testid="stSidebar"] {
        background-color: #111827;
    }

    p, h1, h2, h3, h4, h5 {
        color: #FAFAFA !important;
    }

    div[data-testid="stMarkdownContainer"] {
        color: #FAFAFA !important;
    }
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
    st.markdown(
        "A system that automatically categorizes news articles into topics like World, Science/Technology, Sports and Business."
    )

    st.markdown("### How to use this app:")
    st.markdown("""
    1. Type or paste a news article into the text box below.
    2. Click **Predict Category**.
    3. Explore the dashboard to see the prediction, confidence scores and training analytics.
    """)
    st.divider()


def draw_comparison_dashboard(bert_probabilities, lr_probabilities, bert_pred, lr_pred):
    df_bert = pd.DataFrame({
        "Category": list(bert_probabilities.keys()),
        "Confidence": list(bert_probabilities.values()),
        "Model": "BERT"
    })
    
    df_lr = pd.DataFrame({
        "Category": list(lr_probabilities.keys()),
        "Confidence": list(lr_probabilities.values()),
        "Model": "Linear Regression"
    })
    
    df_combined = pd.concat([df_bert, df_lr])

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 1️⃣ BERT Category Confidence")
        fig1 = px.bar(df_bert, x="Confidence", y="Category", orientation="h",
                     color="Category", text_auto=".2%", color_discrete_sequence=px.colors.qualitative.Pastel)
        fig1.update_layout(xaxis_tickformat="%", showlegend=False, height=260,
                           paper_bgcolor="#0E1117", plot_bgcolor="#0E1117", font_color="white")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.markdown("#### 2️⃣ Linear Regression Category Confidence")
        fig2 = px.bar(df_lr, x="Confidence", y="Category", orientation="h",
                     color="Category", text_auto=".2%", color_discrete_sequence=px.colors.qualitative.Safe)
        fig2.update_layout(xaxis_tickformat="%", showlegend=False, height=260,
                           paper_bgcolor="#0E1117", plot_bgcolor="#0E1117", font_color="white")
        st.plotly_chart(fig2, use_container_width=True)

    st.write("---")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("#### 3️⃣ Side-by-Side Category Distribution Comparison")
        fig3 = px.bar(df_combined, x="Category", y="Confidence", color="Model",
                     barmode="group", text_auto=".1%")
        fig3.update_layout(yaxis_tickformat="%", height=300,
                           paper_bgcolor="#0E1117", plot_bgcolor="#0E1117", font_color="white")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown("#### 4️⃣ Final Decision Confidence Margin")
        
        bert_top_conf = bert_probabilities.get(bert_pred, 0.0)
        lr_top_conf = lr_probabilities.get(lr_pred, 0.0)
        
        fig4 = go.Figure()
        
        fig4.add_trace(go.Indicator(
            mode = "gauge+number",
            value = bert_top_conf * 100,
            domain = {'x': [0, 0.45], 'y': [0, 1]},
            title = {'text': f"BERT: {bert_pred}", 'font': {'size': 14}},
            gauge = {'bar': {'color': "#5dfdcb"}, 'axis': {'range': [0, 100]}}
        ))
        
        fig4.add_trace(go.Indicator(
            mode = "gauge+number",
            value = lr_top_conf * 100,
            domain = {'x': [0.55, 1], 'y': [0, 1]},
            title = {'text': f"LR: {lr_pred}", 'font': {'size': 14}},
            gauge = {'bar': {'color': "#fe5f55"}, 'axis': {'range': [0, 100]}}
        ))
        
        fig4.update_layout(height=280, paper_bgcolor="#0E1117", font_color="white")
        st.plotly_chart(fig4, use_container_width=True)


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