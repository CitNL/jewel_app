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


#@st.cache(allow_output_mutation=True)
def read_file_from_s3(bucket_name, file_key):

    s3_client = boto3.client('s3')
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        st.write('File found and OKKKKKK!!!!!')

    except Exception as e:
        print("Error accessing S3:", e)
        return None
    
    content=response['Body'].read().decode('utf-8')
    st.write('File loaded successfully')
    print('file loaded')
    return content

# Streamlit App interface
st.title("Jewel Classification App")
st.write("Upload an image to classify the jewel.")

if st.button("Load File"):
        bucket_name = 'ndl-sandbox'
        file_key = 'jewel-classifier/categories_test.txt'
        try:
            content = read_file_from_s3(bucket_name, file_key)
            if content is not None:
                # Display the file content
                st.text_area("File Content", content, height=400)
            
                # Split into categories if content exists
                categories = content.split('\n')
                st.write(categories)  # Show the first 10 items from the list
            else:
                st.error("Failed to load content from the file.")

        except Exception as e:
            st.error(f"Error: {e}")


    




# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    