import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# --- Configuration --- #
MODEL_PATH = 'brain_tumor_detection_model.h5'
IMAGE_SIZE = (224, 224)  # Must match the input size of the trained model
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']  # Must match training order

# --- Load the Model --- #
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Error: Model file not found at {MODEL_PATH}. Please ensure it's in the same directory.")
        return None
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


model = load_model()

# --- Prediction Function --- #
def predict_image(image, model):
    if model is None:
        return "Model not loaded.", 0.0

    # Preprocess the image
    img = image.resize(IMAGE_SIZE)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class = CLASS_NAMES[predicted_class_index]
    confidence = np.max(predictions, axis=1)[0]

    return predicted_class, confidence


# --- Streamlit App --- #
st.title("Brain Tumor Detection from MRI Images")
st.write("Upload an MRI image to predict if a tumor is present and its type.")

if model is None:
    st.warning("Model could not be loaded. Please check the MODEL_PATH and ensure the model file exists.")
else:
    uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded MRI Image", use_column_width=True)
        st.write("")

        # Prediction
        with st.spinner("Analyzing image..."):
            predicted_class, confidence = predict_image(image, model)
            st.success("Analysis Complete!")

        st.subheader("Prediction:")

        if predicted_class == "notumor":
            st.write(f"The model predicts: **No Tumor** with {confidence:.2f} confidence.")
        else:
            st.write(
                f"The model predicts: **{predicted_class.replace('_', ' ').title()} Tumor** with {confidence:.2f} confidence."
            )

        st.write("--- DISCLAIMER ---")
        st.info(
            "This application is for educational and demonstrative purposes only and should NOT be used for medical diagnosis. "
            "Always consult with a qualified medical professional for any health concerns."
        )