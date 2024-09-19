import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os
import boto3



# Load your trained model
# @st.cache(allow_output_mutation=True)
# def load_trained_model():
#     model = load_model("jewel_classifier_resnet.h5")
#     return model

@st.cache(allow_output_mutation=True)
def load_trained_model_from_S3():

    s3_client = boto3.client('s3')

    bucket_name = 'ndl-sandbox'
    jewel_model = 'jewel-classifier/jewel_classifier_resnet.h5'

    try:
        vector_obj = s3_client.get_object(Bucket=bucket_name, Key=jewel_model)

    except Exception as e:
        print("Error accessing S3:", e)
        return None

model = load_trained_model_from_S3()

# Define a function to preprocess the uploaded image
def preprocess_image(image, target_size):
    # Resize and convert image to array
    image = ImageOps.fit(image, target_size, Image.ANTIALIAS)
    image = np.asarray(image)
    # Normalize image to [0, 1] range
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

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
    
    # Preprocess the image
    processed_image = preprocess_image(image, target_size=(224, 224))  # adjust target_size based on your model's input size
    
    # Make predictions
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions[0])
    
    # Display the result
    st.write(f"Prediction: {categories[predicted_class]}")
    st.write(f"Confidence: {np.max(predictions[0]) * 100:.2f}%")
