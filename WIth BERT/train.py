# -*- coding: utf-8 -*-
"""
Created on Sat Mar  8 21:12:54 2025

@author: vicky
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split

# ✅ Load dataset
df = pd.read_csv("data/IMDB.csv")
df['sentiment'] = df['sentiment'].map({"positive": 1, "negative": 0})  # Convert labels to binary (1/0)

# ✅ Load BERT Tokenizer
MODEL_NAME = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# ✅ Tokenize and Encode the dataset
def encode_texts(texts, tokenizer, max_length=128):
    return tokenizer(
        texts.tolist(),
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="tf"
    )

encoded_data = encode_texts(df["review"], tokenizer)
X = {
    "input_ids": encoded_data["input_ids"],
    "attention_mask": encoded_data["attention_mask"]
}
y = np.array(df["sentiment"])

# ✅ Split into Train & Test
X_train, X_test, y_train, y_test = train_test_split(X["input_ids"], y, test_size=0.2, random_state=42)

# ✅ Load BERT Model for Classification
bert_model = TFBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# ✅ Compile Model
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
bert_model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

# ✅ Train Model
bert_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=8)

# ✅ Save Model & Tokenizer
bert_model.save_pretrained("model/bert_sentiment")
tokenizer.save_pretrained("model/bert_tokenizer")

print("✅ Model training complete & saved!")
