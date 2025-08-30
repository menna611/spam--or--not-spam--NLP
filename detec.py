import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("spam.csv", encoding="latin-1")
    df = df[['v1','v2']]
    df.columns = ['label','message']
    df['label'] = df['label'].map({'ham':0, 'spam':1})
    return df

# Preprocess and train TensorFlow model
@st.cache_resource
def train_model():
    df = load_data()
    X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train_tfidf.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train_tfidf.toarray(), y_train, epochs=3, batch_size=32, verbose=0)

    return vectorizer, model

vectorizer, tf_model = train_model()

# Streamlit UI
st.set_page_config(page_title="Spam Classifier", page_icon="📩", layout="centered")

st.markdown("""
    <style>
        .main-title {text-align: center; font-size: 36px; color: #4CAF50; font-weight: bold;}
        .footer {text-align: center; font-size: 14px; color: #888; margin-top: 40px;}
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>📩 Spam Message Classifier (TensorFlow)</div>", unsafe_allow_html=True)

st.write("Type any message below and let the TensorFlow model decide if it's **Spam** or **Not Spam**.")

msg = st.text_area("✍️ Enter a message to classify", height=120, placeholder="e.g. Congratulations! You won a free iPhone 🎉")

if st.button("🚀 Predict", use_container_width=True):
    if msg.strip() == "":
        st.warning("⚠️ Please enter a message first.")
    else:
        X_input = vectorizer.transform([msg])
        pred = (tf_model.predict(X_input.toarray()) > 0.5).astype(int)[0][0]

        if pred == 1:
            st.error("🚨 This looks like **Spam**!")
        else:
            st.success("✅ This looks **Safe (Not Spam)**")

st.markdown("<div class='footer'>Built with ❤️ using Streamlit & TensorFlow</div>", unsafe_allow_html=True)


