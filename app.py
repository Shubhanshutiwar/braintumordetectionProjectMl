import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import requests

# --- Configuration --- #
MODEL_PATH = 'brain_tumor_detection_model.h5'
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# --- Page Config --- #
st.set_page_config(page_title="Brain Tumor Detection AI", page_icon="🧠")

# --- Download Model from GitHub Release --- #
@st.cache_resource
def download_model_from_release():
    if not os.path.exists(MODEL_PATH):
        try:
            # Update this URL if your GitHub release link changes
            url = "https://github.com/Shubhanshutiwar/braintumordetectionProjectMl/releases/download/V1/brain_tumor_detection_model.h5"
            
            with st.status("Downloading AI model (this may take a minute)...", expanded=True) as status:
                response = requests.get(url, stream=True, timeout=300)
                response.raise_for_status()
                
                with open(MODEL_PATH, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                status.update(label="Model downloaded successfully!", state="complete", expanded=False)
        except Exception as e:
            st.error(f"Error downloading model: {e}")
            return False
    return True

# --- Load the Model --- #
@st.cache_resource
def load_model_file():
    if not download_model_from_release():
        return None
    
    if not os.path.exists(MODEL_PATH):
        st.error(f"Error: Model file not found at {MODEL_PATH}")
        return None
    
    try:
        # We load with compile=False to avoid issues with custom optimizers/losses
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- Main UI --- #
def main():
    st.title("🧠 Brain Tumor Detection AI")
    st.markdown("---")
    st.write("Upload an MRI scan below, and the AI will analyze it for signs of tumors.")

    # 1. Initialize Model
    model = load_model_file()

    if model:
        # 2. File Uploader
        uploaded_file = st.file_uploader("Choose an MRI image (JPG, PNG)...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Create two columns for a clean look
            col1, col2 = st.columns(2)

            with col1:
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption='Uploaded MRI Scan', use_container_width=True)

            with col2:
                if st.button("🔍 Run Analysis"):
                    with st.spinner("Analyzing scan..."):
                        # Preprocessing
                        img = image.resize(IMAGE_SIZE)
                        img_array = np.array(img) / 255.0  # Normalization
                        img_array = np.expand_dims(img_array, axis=0)

                        # Prediction
                        predictions = model.predict(img_array)
                        
                        # Use Softmax if the model output is raw logits
                        score = tf.nn.softmax(predictions[0])
                        
                        result_idx = np.argmax(score)
                        result_name = CLASS_NAMES[result_idx]
                        confidence = 100 * np.max(score)

                        # Display Results
                        st.subheader("Result:")
                        if result_name == 'notumor':
                            st.success(f"Prediction: {result_name.upper()}")
                        else:
                            st.warning(f"Prediction: {result_name.upper()}")
                        
                        st.progress(int(confidence))
                        st.write(f"Confidence Score: **{confidence:.2f}%**")
    else:
        st.error("Model system is offline. Check logs for details.")

if __name__ == "__main__":
    main()
