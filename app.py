import streamlit as st
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import io

# Load your trained VAE model
model = load_model(r'C:\Users\Owner\Desktop\Deeplearning\vae_model.h5')

# Function to encode the keyword into a numerical format
def encode_keyword(keyword, vocab):
    encoded = np.zeros(len(vocab))
    if keyword in vocab:
        encoded[vocab.index(keyword)] = 1
    return encoded

# Example vocabulary
vocab = ['happy', 'sad', 'angry', 'surprised', 'neutral']

# Streamlit app
st.title("Image Generation using VAE")

keyword = st.selectbox('Select a keyword:', vocab)
generate_button = st.button('Generate Image')

if generate_button:
    condition = encode_keyword(keyword, vocab)

    # Generate random noise
    noise = np.random.normal(0, 1, (1, 100))  # Adjust the dimensions according to your model input

    # Combine noise and condition
    input_vector = np.concatenate([noise, condition.reshape(1, -1)], axis=1)