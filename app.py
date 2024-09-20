import streamlit as st
import numpy as np
import os
from PIL import Image, ImageOps
import boto3


@st.cache(allow_output_mutation=True)
def load_trained_model_from_S3():

    s3_client = boto3.client('s3')

    bucket_name = 'ndl-sandbox'
    jewel_model = 'jewel-classifier/jewel_classifier_resnet.h5'

    try:
        model = s3_client.get_object(Bucket=bucket_name, Key=jewel_model)

    except Exception as e:
        print("Error accessing S3:", e)
        return None
    return model

model = load_trained_model_from_S3()


@st.cache(allow_output_mutation=True)
def load_file_from_S3():

    s3_client = boto3.client('s3')

    bucket_name = 'ndl-sandbox'
    jewel_model = 'jewel-classifier/categories.txt'

    try:
        file = s3_client.get_object(Bucket=bucket_name, Key=jewel_model)

    except Exception as e:
        print("Error accessing S3:", e)
        return None
    return file['Body'].read().decode('utf-8')


cat_file=load_file_from_S3()

# Create a mapping for model predictions
with open(cat_file, 'r') as f:
    categories = [line.strip() for line in f.readlines()]

# Streamlit App interface
st.title("Jewel Classification App")
st.write("Upload an image to classify the jewel.")
st.write(categories[0])

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    