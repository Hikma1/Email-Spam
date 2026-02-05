import streamlit as st
import pickle

# Load model
with open("spam_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load vectorizer
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

st.title("📧 Spam Message Classifier")

user_input = st.text_area("Enter a message:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message")
    else:
        vec = vectorizer.transform([user_input])
        prediction = model.predict(vec)[0]
        st.success("SPAM 🚫" if prediction == 1 else "NOT SPAM ✅")
