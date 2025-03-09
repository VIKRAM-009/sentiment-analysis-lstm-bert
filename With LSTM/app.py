# -*- coding: utf-8 -*-
"""
Created on Sat Mar  8 21:03:28 2025

@author: vicky
"""

import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ✅ Load trained model & tokenizer
model = tf.keras.models.load_model("model/sentiment_model.h5")
with open("model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# ✅ Streamlit UI
st.title("Sentiment Analysis App")
st.write("Enter a review below to analyze its sentiment.")

text_input = st.text_area("Enter your review:", "")

if st.button("Analyze Sentiment"):
    if text_input.strip():
        sequence = tokenizer.texts_to_sequences([text_input])  # Convert text to numbers
        padded = pad_sequences(sequence, maxlen=200)  # Pad sequence
        prediction = model.predict(padded)[0][0]
        sentiment = "Positive" if prediction > 0.5 else "Negative"
        
        st.subheader("Result")
        st.write(f"**Predicted Sentiment:** {sentiment}")
        st.write(f"**Confidence:** {prediction:.2f}")
    else:
        st.warning("Please enter a valid review.")
