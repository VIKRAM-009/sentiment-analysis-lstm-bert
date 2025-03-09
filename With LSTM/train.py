# -*- coding: utf-8 -*-
"""
Created on Sat Mar  8 20:45:36 2025

@author: vicky
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv(r"C:/Users/vicky/Documents/Bert_sentiment_analyser/Data/IMDB.csv")
df['sentiment'] = df['sentiment'].map({"positive": 1, "negative": 0})

# Tokenization & Padding
MAX_WORDS = 10000
MAX_LEN = 200

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(df['review'])

X = tokenizer.texts_to_sequences(df['review'])
X = pad_sequences(X, maxlen=MAX_LEN)

y = np.array(df['sentiment'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build LSTM Model
model = Sequential([
    Embedding(MAX_WORDS, 128, input_length=MAX_LEN),
    LSTM(64, return_sequences=True),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train Model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=64)

# Save model & tokenizer
model.save("model/sentiment_model.h5")
import pickle
with open("model/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("✅ Model training complete & saved!")
##########################PREDICT.py###############################################
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
