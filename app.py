# Import necessary libraries
import streamlit as st
import joblib  # You can use joblib to load your trained model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd

# Load your pre-trained machine learning model
from keras.src.utils import pad_sequences
from tinycss2 import tokenizer

model = joblib.load('Dga_Classification_lstm_model.h5')  # Replace 'your_model.pkl' with your model file path

# Streamlit UI
st.title('Website Legitimacy Predictor')

# Create a text input field for user input
user_input = st.text_input('Enter a website URL:', 'example.com')

# Create a function to make predictions
def predict_legitimacy(url):
    # You should preprocess the input URL to match your model's requirements
    max_sequence_length = 40
    sequences_unseen = tokenizer.texts_to_sequences([url.lower()])
    X_unseen = pad_sequences(sequences_unseen, maxlen=max_sequence_length)
    # Example preprocessing: extracting features, tokenizing, encoding, etc.
    # Replace this example code with your actual preprocessing steps
    features = [X_unseen]  # Replace with your actual feature extraction
    # Convert features to a DataFrame (you may have to adapt this to your model)
    data = pd.DataFrame(features, columns=['url'])
    prediction = model.predict(data)  # Make predictions using your model
    return prediction

# Create a button to trigger predictions
if st.button('Predict'):
    prediction = predict_legitimacy(user_input)
    if prediction[0] == 1:
        st.success('This website is legit.')
    else:
        st.error('This website is a DGA.')

# Optionally, add more content to your Streamlit app, such as explanations or visuals.
