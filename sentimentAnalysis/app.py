import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Title
st.title("Emotion Detection Using NLP")
st.write("Enter text to predict sentiment")

# User input
user_input = st.text_area("Enter Review/Text")

# Predict button
if st.button("Predict"):

    # Transform text
    transformed_text = vectorizer.transform([user_input])

    # Prediction
    prediction = model.predict(transformed_text)[0]

    emotion_labels = {
        0: "Sadness 😢",
        1: "Joy 😊",
        2: "Love ❤️",
        3: "Anger 😠",
        4: "Fear 😨",
        5: "Surprise 😲"
    }

    st.success(f"Predicted Emotion: {emotion_labels[prediction]}")