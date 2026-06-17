import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
import os


def setup_page():

    with st.sidebar:
        st.header("📰 Project Details")

        st.markdown("**Group Name:** R&R Tech Team")

        st.divider()

        st.markdown("### 👥 Group Members")
        st.markdown("- **Virocchan A/L S.Elankoven** *(A24AI0092)*")
        st.markdown("- **Ryan Ang Jun Yau** *(A24AI0119)*")
        st.markdown("- **Muhammad Iman Hakimi Bin Abu Supian** *(A24AI0113)*")

        st.divider()

        st.caption(
            "🎓 Universiti Teknologi Malaysia Kuala Lumpur"
        )

    st.title("📰 R&R News Categorizer - Multilanguage")

    st.markdown(
        """
A system that automatically categorizes news articles into topics like:

- 🌍 World
- 💼 Business
- 🔬 Science/Technology
- ⚽ Sports
"""
    )

    st.markdown("### How to use")

    st.markdown("""
1. Paste a news article.
2. Click **Predict Category**.
3. Review confidence scores.
4. Explore the analytics dashboard.
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
        showlegend=False,
        height=300,
        xaxis_tickformat=".0%",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font_color="white"
    )

    st.plotly_chart(fig, use_container_width=True)


def show_image_dashboard(image_dir):

    images = [
        ("distribution.png", "1. Dataset Distribution"),
        ("loss_curve.png", "2. Training Loss Curve"),
        ("wordcloud.png", "3. Global Word Cloud"),
        ("confusion_matrix.png", "4. Confusion Matrix")
    ]

    cols = st.columns(2)

    for i, (filename, title) in enumerate(images):

        col = cols[i % 2]

        with col:
            st.markdown(f"#### {title}")

            path = os.path.join(image_dir, filename)

            if os.path.exists(path):
                image = Image.open(path)
                st.image(image, use_container_width=True)
            else:
                st.warning(f"{filename} not found.")