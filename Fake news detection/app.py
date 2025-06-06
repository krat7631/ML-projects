
import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')

# Load the saved model and vectorizer
model = pickle.load(open("logistic_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Streamlit App UI
st.title("📰 Fake News Detection")
st.write("Enter a news article below and click **Detect** to see if it's real or fake.")

user_input = st.text_area("📝 Enter News Article Text Here", height=200)

if st.button("Detect"):
    if user_input.strip() == "":
        st.warning("Please enter some text to classify.")
    else:
        input_tfidf = vectorizer.transform([user_input])
        prediction = model.predict(input_tfidf)[0]
        confidence = model.predict_proba(input_tfidf)[0].max() * 100

        if prediction == 1:
            st.success(f"✅ Real News ({confidence:.2f}% confidence)")
        else:
            st.error(f"🚫 Fake News ({confidence:.2f}% confidence)")
