# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 08:32:19 2025

@author: vicky
"""

import numpy as np
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf

# ✅ Load Model & Tokenizer
MODEL_PATH = "model/bert_sentiment"
TOKENIZER_PATH = "model/bert_tokenizer"

tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)
model = TFBertForSequenceClassification.from_pretrained(MODEL_PATH)

# ✅ Function to Predict Sentiment
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="tf", padding="max_length", truncation=True, max_length=128)
    logits = model(**inputs).logits
    prediction = tf.nn.softmax(logits)[0]
    label = "Positive" if tf.argmax(prediction) == 1 else "Negative"
    confidence = prediction[tf.argmax(prediction)].numpy()

    return f"Review: {text}\nPredicted Sentiment: {label} (Confidence: {confidence:.2f})"

# ✅ Get user input
if __name__ == "__main__":
    while True:
        text = input("\nEnter a review (or type 'exit' to quit): ")
        if text.lower() == "exit":
            break
        print(predict_sentiment(text))
