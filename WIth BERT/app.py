# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 08:34:18 2025

@author: vicky
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification

# ✅ Load Model & Tokenizer
MODEL_PATH = "model/bert_sentiment"
TOKENIZER_PATH = "model/bert_tokenizer"

tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)
model = TFBertForSequenceClassification.from_pretrained(MODEL_PATH)

# ✅ Streamlit UI
st.title("Sentiment Analysis with BERT")
st.write("Enter a review below to analyze its sentiment.")

text_input = st.text_area("Enter your review:", "")

if st.button("Analyze Sentiment"):
    if text_input.strip():
        inputs = tokenizer(text_input, return_tensors="tf", padding="max_length", truncation=True, max_length=128)
        logits = model(**inputs).logits
        prediction = tf.nn.softmax(logits)[0]
        label = "Positive" if tf.argmax(prediction) == 1 else "Negative"
        confidence = prediction[tf.argmax(prediction)].numpy()
        
        st.subheader("Result")
        st.write(f"**Predicted Sentiment:** {label}")
        st.write(f"**Confidence:** {confidence:.2f}")
    else:
        st.warning("Please enter a valid review.")
