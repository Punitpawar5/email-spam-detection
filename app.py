import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model and vectorizer from pickle files
with open('spam_classifier_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Streamlit app interface
st.title("Email Spam Detection App")

st.write("""
### Enter the email content below:
""")

# Input text box for the user to enter email text
input_email = st.text_area("Email content", height=200)

# When the 'Predict' button is clicked
if st.button("Predict"):
    # Preprocess and transform the input using the vectorizer
    email_tfidf = vectorizer.transform([input_email])

    # Make the prediction
    prediction = model.predict(email_tfidf)

    # Display the result
    if prediction == 1:
        st.error("This email is predicted to be **SPAM**.")
    else:
        st.success("This email is predicted to be **HAM** (not spam).")

