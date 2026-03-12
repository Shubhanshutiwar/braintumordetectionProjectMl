import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import requests
from pathlib import Path

# --- Configuration --- #
MODEL_PATH = 'brain_tumor_detection_model.h5'
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# --- Download Model from GitHub Release --- #
@st.cache_resource
def download_model_from_release():
    if not os.path.exists(MODEL_PATH):
        try:
            with st.spinner("Downloading model from GitHub release..."):
                url = "https://github.com/Shubhanshutiwar/braintumordetectionProjectMl/releases/download/V1/brain_tumor_detection_model.h5"
                response = requests.get(url, timeout=300)
                response.raise_for_status()
                
                with open(MODEL_PATH, 'wb') as f:
                    f.write(response.content)
                st.success("Model downloaded successfully!")
        except Exception as e:
            st.error(f"Error downloading model: {e}")
            return False
    return True

# --- Load the Model --- #
@st.cache_resource
def load_model():
    if not download_model_from_release():
        return None
    
    if not os.path.exists(MODEL_PATH):
        st.error(f"Error: Model file not found at {MODEL_PATH}")
        return None
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# ... rest of your code remains the same ...
