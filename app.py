import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import os
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load the TFLite model
MODEL_PATH = os.getenv('MODEL_PATH', 'best_model2.tflite')
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

CLASSES = ['Blackheads', 'Cyst', 'Papules', 'Pustules', 'Whiteheads']

# Function to preprocess image
def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return img_array

# Function to make predictions
def predict_image(image_array):
    if image_array.dtype != np.float32:
        image_array = image_array.astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], image_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0]

# Streamlit UI
st.title("Acne Detection System")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    st.write("Processing...")
    processed_image = preprocess_image(image)
    predictions = predict_image(processed_image)
    
    top_prediction_idx = np.argmax(predictions)
    top_prediction = CLASSES[top_prediction_idx]
    confidence = float(predictions[top_prediction_idx])
    
    st.write(f"Prediction: {top_prediction} ({confidence * 100:.2f}% confidence)")
