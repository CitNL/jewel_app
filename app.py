import streamlit as st
import numpy as np
import os
from PIL import Image, ImageOps
import boto3
import tensorflow as tf
from tensorflow.keras.models import load_model

aws_access_key_id = st.secrets["AWS_ACCESS_KEY_ID"]
aws_secret_access_key = st.secrets["AWS_SECRET_ACCESS_KEY"]

@st.cache_resource()
def load_trained_model_from_S3():

    s3_client = boto3.client('s3',
                             aws_access_key_id=aws_access_key_id,
                             aws_secret_access_key=aws_secret_access_key)
    
    bucket_name = 'ndl-sandbox'
    jewel_model = 'jewel-classifier/jewel_classifier_resnet.h5'
    # Temporary local path
    local_path = 'jewel_model.h5'  # Adjust based on your environment #'/tmp/jewel_model.h5'
    try:
        #response = s3_client.get_object(Bucket=bucket_name, Key=jewel_model)
        s3_client.download_file(bucket_name, jewel_model, local_path)
        #st.write('File found and OKKKKKK!!!!!')
        model = load_model(local_path)
        st.write('model loaded')

    except Exception as e:
        st.error(f"Error: {e}")
        return None
    
    return model

# if st.button("Load Model"):
#     model = load_trained_model_from_S3()
#     if model:
#         st.write('model loaded')

model = load_trained_model_from_S3()

@st.cache_resource()
def read_file_from_s3(bucket_name, file_key):

    s3_client = boto3.client('s3',
                             aws_access_key_id=aws_access_key_id,
                             aws_secret_access_key=aws_secret_access_key)
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        

    except Exception as e:
        print("Error accessing S3:", e)
        return None
    
    content=response['Body'].read().decode('utf-8')
    
    print('file loaded')
    return content


@st.cache_data()
def load_categories():
        bucket_name = 'ndl-sandbox'
        file_key = 'jewel-classifier/categories.txt'
        try:
            content = read_file_from_s3(bucket_name, file_key)
            if content is not None:           
                # Split into categories if content exists
                categories = content.split('\n')
                st.write('Category labels uploaded')
            else:
                
                st.error("Failed to load category labels")
                return None

        except Exception as e:
            st.error(f"Error: {e}")
            return None
        return categories

categories=load_categories()

# Define a function to preprocess the uploaded image
def preprocess_image(image, target_size):
    # Resize and convert image to array
    image = ImageOps.fit(image, target_size,Image.Resampling.LANCZOS)
    image = np.asarray(image)
    # Normalize image to [0, 1] range
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image


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
    st.image(processed_image, caption="Processed Image", use_column_width=True)
    # Make predictions
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions[0])
    
    # Display the result
    st.write(f"Prediction: {categories[predicted_class]}")
    st.write(f"Confidence: {np.max(predictions[0]) * 100:.2f}%")