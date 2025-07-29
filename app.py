import streamlit as st
import joblib
import numpy as np

# 🎯 Load model, vectorizer, and label encoder
model = joblib.load("optimized_logistic_model.pkl")
vectorizer = joblib.load("bow_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# 🌟 App Title
st.set_page_config(page_title="Emotion Detector", layout="centered")
st.title("🧠 Emotion Detector - Logistic Regression (Bag of Words)")
st.markdown(
    "Enter a sentence to detect the underlying emotion using a trained ML model.")

# 📝 Text Input
text = st.text_area("Enter your text here:")

# 🧠 Prediction
if st.button("Predict Emotion"):
    if not text.strip():
        st.warning("Please enter some text!")
    else:
        transformed_text = vectorizer.transform([text])
        prediction = model.predict(transformed_text)[0]
        emotion = label_encoder.inverse_transform([prediction])[0]
        st.success(f"**Predicted Emotion:** {emotion.upper()}")

# 📌 Footer
st.markdown(
    """
    <hr style="border:0.5px solid gray">
    <div style="text-align:center">
        Built with ❤️ using Logistic Regression, Bag of Words, and Streamlit.
    </div>
    """,
    unsafe_allow_html=True
)
