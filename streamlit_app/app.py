import streamlit as st
import numpy as np
import cv2
import tensorflow as tf

# Load your pre-trained model
model_load = tf.keras.models.load_model('VGG_freeze.h5')

# Define your class names
class_names = ["angry","happy","ahegao","sad","neutral","surprise"]     # Replace with your actual class names

def process_image(image):
    image_size = (64, 64)
    img_data = cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)
    img_data = img_data / 255.0
    img_data = np.array([img_data])
    return img_data

st.title("CNN Image Classifier")
st.write("Upload an image and the model will predict its class.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Process the image
    img_data = process_image(image)
    st.write(f"Processed image shape: {img_data.shape}")

    # Make prediction
    prediction = model_load.predict(img_data)
    predicted_class = class_names[np.argmax(prediction)]

    st.write(f"Predicted class: {predicted_class}")
