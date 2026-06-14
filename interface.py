import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
import os

def setup_page():
    st.set_page_config(page_title="R&R News Categorizer", layout="wide")
    
    # --- SIDEBAR: Professional Group Details ---
    with st.sidebar:
        st.header("📰 Project Details")
        
        # You can change "R&R Tech Team" to your actual group name!
        st.markdown("**Group Name:** R&R Tech Team") 
        
        st.divider()
        
        st.markdown("### 👥 Group Members")
        # Using two spaces at the end of the name creates a tight, professional line break
        st.markdown("- **Virocchan A/L S.Elankoven** \n  *(A24AI0092)*")
        st.markdown("- **Ryan Ang Jun Yau** \n  *(A24AI0119)*")
        st.markdown("- **Muhammad Iman Hakimi Bin Abu Supian** \n  *(A24AI0113)*")
        
        st.divider()
        st.caption("🎓 Universiti Teknologi Malaysia Kuala Lumpur")
    # -------------------------------------------

    # --- MAIN PAGE CONTENT ---
    st.title("📰 R&R News Categorizer")
    
    st.markdown("### What problem are we solving?")
    st.markdown("With thousands of news articles generated online every day, it is impossible to read and categorize them all manually. Our NLP system automatically reads news snippets and predicts the category instantly.")
    
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
    
    fig = px.bar(df, x="Confidence", y="Category", orientation='h', color="Category", text_auto='.2%')
    fig.update_layout(xaxis_tickformat='%', showlegend=False, height=280)
    st.plotly_chart(fig, use_container_width=True)

def show_image_dashboard(image_dir):
    col1, col2 = st.columns(2)
    
    def load_img(column, filename, title):
        column.markdown(f"#### {title}")
        path = os.path.join(image_dir, filename)
        if os.path.exists(path):
            # Open the image and resize it while keeping the Colab transparency intact
            img = Image.open(path).resize((600, 400))
            column.image(img, use_container_width=True)
        else:
            column.warning(f"Image {filename} not found.")
            
    with col1:
        load_img(col1, "distribution.png", "1. Dataset Distribution")
        st.write("") # Spacer
        load_img(col1, "wordcloud.png", "3. Global Word Cloud")

    with col2:
        load_img(col2, "loss_curve.png", "2. Training Loss Curve")
        st.write("") # Spacer
        load_img(col2, "confusion_matrix.png", "4. Confusion Matrix")