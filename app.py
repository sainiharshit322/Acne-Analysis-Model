import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Load the TFLite model
MODEL_PATH = os.getenv('MODEL_PATH', 'best_model2.tflite')
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Acne classes with precautions & cures
ACNE_CLASSES = ['Blackheads', 'Cyst', 'Papules', 'Pustules', 'Whiteheads']

ACNE_INFO = {
    'Blackheads': {
        'precautions': ["Wash your face twice daily", "Use non-comedogenic skincare", "Avoid excessive oil-based products"],
        'cure': "Use salicylic acid or benzoyl peroxide products."
    },
    'Cyst': {
        'precautions': ["Avoid picking or popping", "Keep skin clean", "Use mild, fragrance-free products"],
        'cure': "Consult a dermatologist for prescription medication."
    },
    'Papules': {
        'precautions': ["Use gentle cleansers", "Avoid harsh scrubbing", "Apply tea tree oil"],
        'cure': "Use benzoyl peroxide or topical retinoids."
    },
    'Pustules': {
        'precautions': ["Avoid touching affected areas", "Use lightweight, oil-free products", "Change pillowcases regularly"],
        'cure': "Apply over-the-counter acne creams containing sulfur or salicylic acid."
    },
    'Whiteheads': {
        'precautions': ["Exfoliate regularly with AHA/BHA", "Use non-comedogenic moisturizer", "Wash makeup brushes weekly"],
        'cure': "Use topical retinoids or niacinamide-based products."
    }
}

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
    return output_data[0]  # Returns probabilities for all classes

# Function to plot a horizontal bar chart
def plot_acne_probabilities(probabilities):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(ACNE_CLASSES, probabilities * 100, color=['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF6666'])
    
    ax.set_xlabel("Confidence (%)")
    ax.set_title("Acne Type Prediction Confidence")
    ax.invert_yaxis()  # Highest confidence on top

    for i, v in enumerate(probabilities * 100):
        ax.text(v + 1, i, f"{v:.2f}%", va='center')

    st.pyplot(fig)

# Streamlit UI
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Acne Detection", "Products"])

if page == "Acne Detection":
    st.title("Acne Detection System")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)

        st.write("Processing...")
        processed_image = preprocess_image(image)
        probabilities = predict_image(processed_image)

        # Show bar chart of all acne class probabilities
        plot_acne_probabilities(probabilities)

        # Get top prediction
        top_prediction_idx = np.argmax(probabilities)
        top_prediction = ACNE_CLASSES[top_prediction_idx]
        confidence = float(probabilities[top_prediction_idx])

        st.write(f"**Top Prediction: {top_prediction} ({confidence * 100:.2f}% confidence)**")

        # Display precautions and cures
        st.subheader("Precautions & Cure")
        st.write("**Precautions:**")
        for precaution in ACNE_INFO[top_prediction]['precautions']:
            st.write(f"- {precaution}")

        st.write("**Cure:**")
        st.write(ACNE_INFO[top_prediction]['cure'])

elif page == "Products":
    st.title("Recommended Products")
    
    with open("templates/products.html", "r", encoding="utf-8") as file:
        html_code = file.read()
        st.components.v1.html(html_code, height=600, scrolling=True)
