# -*- coding: utf-8 -*-
"""
Created on Sat Mar  8 21:08:27 2025

@author: vicky
"""

import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model & tokenizer
model = tf.keras.models.load_model("model/sentiment_model.h5")
with open("model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Function to predict sentiment
def predict_sentiment(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=200)
    prediction = model.predict(padded)[0][0]
    
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    return f"Review: {text}\nPredicted Sentiment: {sentiment} (Confidence: {prediction:.2f})"

# Test user input
if __name__ == "__main__":
    while True:
        text = input("\nEnter a review (or type 'exit' to quit): ")
        if text.lower() == "exit":
            break
        print(predict_sentiment(text))
