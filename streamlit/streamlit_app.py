#!/usr/bin/env python3
"""
Streamlit Frontend for London City Classification

Real-time image classification with drag-and-drop interface.
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import sys
import os
from pathlib import Path

# Add models to path
sys.path.append('.')
from models.primary_model.resnet_cbam_mlp import ResNet50_CBAM_MLP

# Page config
st.set_page_config(
    page_title="London vs London Classifier",
    page_icon="",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load the trained model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet50_CBAM_MLP().to(device)
    
    model_path = "scripts/trained_model.pth"
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model, device
    else:
        st.error("Trained model not found! Please train the model first.")
        return None, device

def preprocess_image(image):
    """Preprocess image for model input."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    
    # Convert to PIL if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Apply transforms
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

def predict_image(model, image_tensor, device):
    """Make prediction on image."""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        probability = torch.sigmoid(output).item()
        
        # Class mapping: 0 = London_ON, 1 = London_UK
        prediction = "London, UK" if probability > 0.5 else "London, Ontario"
        confidence = probability if probability > 0.5 else 1 - probability
        
        return prediction, confidence, probability

def main():
    # Header
    st.title("London vs London City Classifier")
    st.markdown("Upload a street view image to classify whether it's from **London, UK** or **London, Ontario, Canada**")
    
    # Load model
    model, device = load_model()
    if model is None:
        st.stop()
    
    # Sidebar
    st.sidebar.header("About")
    st.sidebar.markdown("""
    This AI model distinguishes between Google Street View images from:
    
    🇬🇧 **London, UK** - The capital of England
    🇨🇦 **London, Ontario** - A city in Canada
    
    The model was trained on thousands of street view images using a ResNet50 architecture with attention mechanisms.
    """)
    
    st.sidebar.header("Model Info")
    st.sidebar.markdown(f"**Device**: {device}")
    st.sidebar.markdown("**Architecture**: ResNet50 + CBAM + MLP")
    st.sidebar.markdown("**Input Size**: 224×224 pixels")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a street view image to classify"
        )
        
        # Camera input
        camera_photo = st.camera_input("Or take a photo")
        
        # Use either uploaded file or camera photo
        image_file = uploaded_file if uploaded_file is not None else camera_photo
        
        if image_file is not None:
            # Display uploaded image
            image = Image.open(image_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Preprocess image
            try:
                image_tensor = preprocess_image(image)
                
                # Make prediction
                prediction, confidence, probability = predict_image(model, image_tensor, device)
                
                # Display results
                with col2:
                    st.header("🎯 Classification Results")
                    
                    # Prediction with confidence
                    if prediction == "London, UK":
                        st.success(f"🇬🇧 **{prediction}**")
                    else:
                        st.info(f"🇨🇦 **{prediction}**")
                    
                    # Confidence bar
                    st.metric("Confidence", f"{confidence:.1%}")
                    st.progress(confidence)
                    
                    # Detailed probabilities
                    st.subheader("Detailed Probabilities")
                    
                    uk_prob = probability
                    on_prob = 1 - probability
                    
                    col_uk, col_on = st.columns(2)
                    with col_uk:
                        st.metric("London, UK", f"{uk_prob:.1%}")
                    with col_on:
                        st.metric("London, Ontario", f"{on_prob:.1%}")
                    
                    # Probability bar chart
                    prob_data = {
                        "London, UK": uk_prob,
                        "London, Ontario": on_prob
                    }
                    st.bar_chart(prob_data)
                    
                    # Interpretation
                    st.subheader("💡 Interpretation")
                    if confidence > 0.8:
                        st.success("High confidence prediction - the model is very sure about this classification.")
                    elif confidence > 0.6:
                        st.warning("Moderate confidence prediction - the model is reasonably sure.")
                    else:
                        st.error("Low confidence prediction - the model is uncertain about this classification.")
                    
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Built with Streamlit • Powered by PyTorch • Trained on Google Street View Data</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 