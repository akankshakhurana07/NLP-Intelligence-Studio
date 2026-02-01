import streamlit as st
import nltk
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd

# ======================================================
# STREAMLIT SAFE NLTK SETUP
# ======================================================
NLTK_DATA_DIR = "/tmp/nltk_data"
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DATA_DIR)

@st.cache_resource
def setup_nltk():
    nltk.download("punkt", download_dir=NLTK_DATA_DIR, quiet=True)
    nltk.download("stopwords", download_dir=NLTK_DATA_DIR, quiet=True)
    nltk.download("wordnet", download_dir=NLTK_DATA_DIR, quiet=True)

setup_nltk()

# ======================================================
# PAGE CONFIG (MUST BE FIRST STREAMLIT COMMAND)
# ======================================================
st.set_page_config(
    page_title="NLP Intelligence Studio",
    page_icon="🧠",
    layout="wide"
)

# ======================================================
# STYLING
# ======================================================
st.markdown("""
<style>
body {
    background-color:#020617;
}
.big-title {
    font-size:48px;
    font-weight:800;
    background: linear-gradient(90deg,#22d3ee,#38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# TITLE
# ======================================================
st.markdown('<div class="big-title">🧠 NLP Intelligence Studio</div>', unsafe_allow_html=True)
st.caption("Advanced Natural Language Processing Playground")

# ======================================================
# INPUT
# ======================================================
text = st.text_area(
    "✍️ Enter your text",
    height=200,
    placeholder="Paste or type any English text here..."
)

if text.strip():

    # ---------- TOKENIZATION ----------
    tokens = word_tokenize(text)
    st.subheader("🔹 Tokenization")
    st.write(tokens)

    # ---------- STOPWORDS ----------
    stop_words = set(stopwords.words("english"))
    filtered = [w for w in tokens if w.lower() not in stop_words and w.isalpha()]
    st.subheader("🔹 Stopword Removal")
    st.write(filtered)

    # ---------- LEMMATIZATION ----------
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(w) for w in filtered]
    st.subheader("🔹 Lemmatization")
    st.write(lemmas)

    # ---------- WORD





