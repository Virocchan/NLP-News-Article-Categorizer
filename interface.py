import streamlit as st
import pandas as pd
import plotly.express as px
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
        st.markdown("- **Virocchan A/L S.Elankoven** \n  *(A24AI0092)*")
        st.markdown("- **Ryan Ang Jun Yau** \n  *(A24AI0119)*")
        st.markdown("- **Muhammad Iman Hakimi Bin Abu Supian** \n  *(A24AI0113)*")

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
    3. Explore the dashboard to see the prediction, confidence scores, and training analytics.
    """)
    st.divider()


def draw_confidence_chart(probabilities):
    df = pd.DataFrame({
        "Category": list(probabilities.keys()),
        "Confidence": list(probabilities.values())
    })

    fig = px.bar(
        df,
        x="Confidence",
        y="Category",
        orientation="h",
        color="Category",
        text_auto=".2%"
    )

    fig.update_layout(
        xaxis_tickformat="%",
        showlegend=False,
        height=280,
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font_color="white"
    )

    st.plotly_chart(fig, use_container_width=True)


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