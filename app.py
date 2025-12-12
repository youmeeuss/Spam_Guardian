import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
import os
import time
from nltk.stem.porter import PorterStemmer

application_directory = "/Users/shaguunn/Desktop/SMS_EMAIL_SPAM_Detector_web_app/App"
os.chdir(application_directory)

nltk.download('stopwords', quiet=True)

text_stemmer = PorterStemmer()
english_stopwords = set(stopwords.words('english'))
punctuation_characters = set(string.punctuation)

st.set_page_config(
    page_title="🛡️ Spam Guardian",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

application_styles = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        color: #ffffff;
    }
    
    .main {
        background: transparent;
    }
    
    .title {
        font-family: 'Inter', sans-serif;
        font-size: 2.8rem;
        font-weight: 600;
        text-align: center;
        color: #ffffff;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }
    
    .subtitle {
        font-family: 'Inter', sans-serif;
        text-align: center;
        color: rgba(255, 255, 255, 0.8);
        font-size: 1rem;
        font-weight: 400;
        margin-bottom: 3rem;
    }
    
    .main-container {
        background: rgba(255, 255, 255, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 20px;
        padding: 2.5rem;
        margin: 2rem 0;
        backdrop-filter: blur(20px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    }
    
    .stTextArea textarea {
        background: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 12px !important;
        color: #ffffff !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 15px !important;
        padding: 1rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextArea textarea:focus {
        border-color: rgba(255, 255, 255, 0.5) !important;
        box-shadow: 0 0 0 3px rgba(255, 255, 255, 0.1) !important;
        background: rgba(255, 255, 255, 0.2) !important;
    }
    
    .stButton button {
        background: rgba(255, 255, 255, 0.2) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        transition: all 0.3s ease !important;
        backdrop-filter: blur(10px) !important;
        width: 100% !important;
    }
    
    .stButton button:hover {
        transform: translateY(-2px) !important;
        background: rgba(255, 255, 255, 0.3) !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.2) !important;
    }
    
    .result-spam {
        background: rgba(239, 68, 68, 0.9);
        color: white;
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        font-family: 'Inter', sans-serif;
        font-size: 1.3rem;
        font-weight: 600;
        box-shadow: 0 8px 25px rgba(239, 68, 68, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
    }
    
    .result-safe {
        background: rgba(16, 185, 129, 0.9);
        color: white;
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        font-family: 'Inter', sans-serif;
        font-size: 1.3rem;
        font-weight: 600;
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
    }
    
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .stat-card {
        background: rgba(255, 255, 255, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .stat-card:hover {
        background: rgba(255, 255, 255, 0.25);
        transform: translateY(-2px);
    }
    
    .stat-number {
        font-family: 'Inter', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.5rem;
    }
    
    .stat-label {
        font-family: 'Inter', sans-serif;
        color: rgba(255, 255, 255, 0.8);
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .confidence-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.2);
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        margin-top: 0.5rem;
    }
    
    .footer {
        text-align: center;
        color: rgba(255, 255, 255, 0.7);
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        margin-top: 3rem;
        padding-top: 2rem;
        border-top: 1px solid rgba(255, 255, 255, 0.2);
    }
</style>
"""

st.markdown(application_styles, unsafe_allow_html=True)

def preprocess_message_text(raw_message_content):
    normalized_text = raw_message_content.lower()
    tokenized_words = nltk.word_tokenize(normalized_text)
    
    alphanumeric_tokens = []
    for current_token in tokenized_words:
        if current_token.isalnum():
            alphanumeric_tokens.append(current_token)
    
    filtered_meaningful_words = []
    for meaningful_word in alphanumeric_tokens:
        if meaningful_word not in english_stopwords and meaningful_word not in punctuation_characters:
            filtered_meaningful_words.append(meaningful_word)
    
    stemmed_word_collection = []
    for filtered_word in filtered_meaningful_words:
        stemmed_root_word = text_stemmer.stem(filtered_word)
        stemmed_word_collection.append(stemmed_root_word)
    
    processed_text_result = " ".join(stemmed_word_collection)
    return processed_text_result

vectorizer_model_path = 'vectorizer2.pkl'
classification_model_path = 'model2.pkl'

with open(vectorizer_model_path, 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

with open(classification_model_path, 'rb') as model_file:
    spam_classification_model = pickle.load(model_file)

application_title_html = '<h1 class="title">🛡️ Spam Guardian</h1>'
application_subtitle_html = '<p class="subtitle">Intelligent message classification powered by machine learning</p>'

st.markdown(application_title_html, unsafe_allow_html=True)
st.markdown(application_subtitle_html, unsafe_allow_html=True)

left_column, center_column, right_column = st.columns([1, 3, 1])

with center_column:
    user_message_input = st.text_area(
        "Message Analysis",
        placeholder="Enter your SMS or email message here for analysis...",
        height=120,
        help="Paste any message to check if it's spam or legitimate",
        label_visibility="collapsed"
    )
    
    spacing_html = "<br>"
    st.markdown(spacing_html, unsafe_allow_html=True)
    message_analysis_button = st.button('Analyze Message')

if message_analysis_button and user_message_input:
    analysis_spinner_text = 'Analyzing...'
    processing_delay_seconds = 0.8
    
    with st.spinner(analysis_spinner_text):
        time.sleep(processing_delay_seconds)
        
        preprocessed_message_text = preprocess_message_text(user_message_input)
        vectorized_message_features = tfidf_vectorizer.transform([preprocessed_message_text])
        spam_prediction_result = spam_classification_model.predict(vectorized_message_features)[0]
        prediction_confidence_scores = spam_classification_model.predict_proba(vectorized_message_features)[0]
        maximum_confidence_score = prediction_confidence_scores.max()
    
    result_left_column, result_center_column, result_right_column = st.columns([1, 3, 1])
    
    with result_center_column:
        spam_classification_threshold = 1
        
        if spam_prediction_result == spam_classification_threshold:
            spam_detection_html = f"""
            <div class="result-spam">
                ⚠️ Spam Detected
                <div class="confidence-badge">Confidence: {maximum_confidence_score:.1%}</div>
            </div>
            """
            st.markdown(spam_detection_html, unsafe_allow_html=True)
        else:
            safe_message_html = f"""
            <div class="result-safe">
                ✅ Message is Safe
                <div class="confidence-badge">Confidence: {maximum_confidence_score:.1%}</div>
            </div>
            """
            st.markdown(safe_message_html, unsafe_allow_html=True)

elif message_analysis_button and not user_message_input:
    empty_input_error_message = "Please enter a message to analyze"
    st.error(empty_input_error_message)

project_features_html = """
<div class="stats-grid">
    <div class="stat-card">
        <div class="stat-number">TF-IDF</div>
        <div class="stat-label">Vectorization</div>
        <div style="font-size: 0.75rem; margin-top: 0.5rem; opacity: 0.8;">Converts text into numerical vectors based on word importance and frequency</div>
    </div>
    <div class="stat-card">
        <div class="stat-number">NLP</div>
        <div class="stat-label">Text Processing</div>
        <div style="font-size: 0.75rem; margin-top: 0.5rem; opacity: 0.8;">Tokenizes, filters stopwords, and stems words to extract meaningful features</div>
    </div>
    <div class="stat-card">
        <div class="stat-number">ML</div>
        <div class="stat-label">Classification</div>
        <div style="font-size: 0.75rem; margin-top: 0.5rem; opacity: 0.8;">Uses trained algorithms to predict spam vs legitimate messages with confidence</div>
    </div>
</div>
"""

st.markdown(project_features_html, unsafe_allow_html=True)

application_footer_html = """
<div class="footer">
    Binary classification using supervised learning • Text preprocessing with NLTK • Feature extraction with scikit-learn
</div>
"""

st.markdown(application_footer_html, unsafe_allow_html=True)