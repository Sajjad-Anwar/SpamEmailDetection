import streamlit as st
import joblib
import pandas as pd
import re

def clean_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = text.lower()               # Convert to lowercase
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text

# Load the model and vectorizer
model = joblib.load('spam_detector_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Streamlit app title
st.title("Spam Email Detection System")

# User input for email text
input_text = st.text_area("Enter your email text:")

# Predict button
if st.button("Predict"):
    # Preprocess and vectorize the input
    cleaned_input = clean_text(input_text)
    input_vectorized = vectorizer.transform([cleaned_input])
    
    # Make prediction
    prediction = model.predict(input_vectorized)
    
    # Display the result
    if prediction[0] == 1:
        st.write("🚫 This is a **Spam Email**!")
    else:
        st.write("✅ This is **Not Spam**.")
