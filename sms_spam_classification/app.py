import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
import string

ps = PorterStemmer()

# Load your actual training data
training_data = pd.read_csv('C:/Users/E-TIME/Machine learning project/sms-spam-classifier/spam.csv', encoding='latin1')

# Extract features (X_train) and labels (y_train)
X_train_text = training_data['v2']  # Assuming 'v2' is the text column
y_train = LabelEncoder().fit_transform(training_data['v1'])  # Encode labels

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)

# Train the model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Save the trained model and vectorizer using pickle
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(tfidf_vectorizer, vectorizer_file)

# Load the model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

st.title("Email/SMS Spam Classifier")

input_sms = st.text_input("Enter the message")

if st.button('Predict'):
    # Preprocess input text
    transformed_sms = transform_text(input_sms)
    # Vectorize using TF-IDF
    vector_input = tfidf_vectorizer.transform([transformed_sms])
    # Make prediction
    result = model.predict(vector_input)[0]
    # Display result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
