import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download NLTK resources (if not already downloaded)
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

# Load saved model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Function to preprocess text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    # Filter out non-alphanumeric tokens and stopwords
    filtered_text = [ps.stem(word) for word in text if word.isalnum() and word not in stopwords.words('english')]
    
    return " ".join(filtered_text)

# Custom CSS styles for Streamlit components
custom_styles = """
    <style>
        /* Title style */
        .title-wrapper {
            background-color: #003049;
            padding: 1rem;
            border-radius: 12px;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .title-wrapper h1 {
            color: #ffffff;
            text-align: center;
            font-size: 2.5rem;
            font-weight: bold;
        }
        /* Text area style */
        div.stTextArea textarea {
            height: 150px;
            font-size: 20px;
            border-radius: 8px;
            padding: 0.5rem;
            border: 4px solid #fff;
        }
        /* Predict button style */
        div.stButton button {
            background-color: #e63946;
            color: #ffffff;
            font-size: 18px;
            border-radius: 8px;
            padding: 0.75em 1.5em;
            transition: all 0.3s ease;
        }
        div.stButton button:hover {
            background-color: green;
            color : #fff;
        }
        /* Result message style */
        div[data-baseweb="toast"] {
            animation: fadeInDown 0.5s ease forwards;
        }
        @keyframes fadeInDown {
            0% {
                opacity: 0;
                transform: translateY(-20px);
            }
            100% {
                opacity: 1;
                transform: translateY(0);
            }
        }
    </style>
"""

# Inject custom CSS styles
st.markdown(custom_styles, unsafe_allow_html=True)

# Page title and description
st.title("Message Spam Classifier")
st.markdown("Enter a message below to determine if it's spam or not.", unsafe_allow_html=True)

# Text input for user
input_sms = st.text_area("Enter your message here:")

# Predict button
if st.button('Predict'):
    if input_sms:
        # Preprocess input text
        transformed_sms = transform_text(input_sms)
        
        # Vectorize input using TF-IDF vectorizer
        vector_input = tfidf.transform([transformed_sms])
        
        # Make prediction
        prediction = model.predict(vector_input)[0]
        
        # Display prediction result with animation
        if prediction == 1:
            st.error("This message is classified as Spam.")
        else:
            st.success("This message is classified as Not Spam.")
    else:
        st.warning("Please enter a message to predict.")
