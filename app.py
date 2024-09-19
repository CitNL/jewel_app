import streamlit as st
import numpy as np
import os
from PIL import Image, ImageOps


# Load your trained model
# @st.cache(allow_output_mutation=True)
# def load_trained_model():
#     model = load_model("jewel_classifier_resnet.h5")
#     return model


# Create a mapping for model predictions
with open('categories.txt', 'r') as f:
    categories = [line.strip() for line in f.readlines()]

# Streamlit App interface
st.title("Jewel Classification App")
st.write("Upload an image to classify the jewel.")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    