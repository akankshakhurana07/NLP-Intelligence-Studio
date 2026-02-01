import streamlit as st

# ================= PAGE CONFIG (MUST BE FIRST) =================
st.set_page_config(
    page_title="NLP Intelligence Studio",
    layout="wide",
    page_icon="🧠"
)

# ================= IMPORTS =================
import nltk
import spacy
import re
import os
import sys
import subprocess
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter
from nltk.tokenize import (
    sent_tokenize, word_tokenize,
    blankline_tokenize, WhitespaceTokenizer,
    WordPunctTokenizer
)
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
from wordcloud import WordCloud

# ================= NLTK SETUP =================
NLTK_DIR = "/tmp/nltk_data"
os.makedirs(NLTK_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DIR)

@st.cache_resource
def setup_nltk():
    nltk.download("punkt", download_dir=NLTK_DIR)
    nltk.download("stopwords", download_dir=NLTK_DIR)
    nltk.download("wordnet", download_dir=NLTK_DIR)
    nltk.download("averaged_perceptron_tagger", download_dir=NLTK_DIR)

setup_nltk()

# ================= SPACY SETUP =================
@st.cache_resource
def load_spacy():
    try:
        return spacy.load("en_core_web_sm")
    except:
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", "en_core_web_sm"]
        )
        return spacy.load("en_core_web_sm")

nlp = load_spacy()

# ================= UI STYLING =================
st.markdown("""
<style>
body {background-color:#0f172a;}
h1,h2,h3 {color:#38bdf8;}
.card {
    background:#020617;
    padding:18px;
    border-radius:14px;
    color:white;
    box-shadow:0 0 20px rgba(56,189,248,0.35);
}
.metric {
    font-size:30px;
    font-weight:800;
    color:#22d3ee;
}
</style>
""", unsafe_allow_html=True)

# ================= TITLE =================
st.title("🧠 NLP Intelligence Studio — ENGINEER LEVEL")
st.caption("NLU • Linguistic Analysis • Feature Engineering • Semantic Modeling")

text = st.text_area("✍️ Paste Paragraph", height=220)

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# ================= TEXT STATS =================
if text.strip():
    words = word_tokenize(text)
    sentences = sent_tokenize(text)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='card'>Words<br><span class='metric'>{len(words)}</span></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='card'>Sentences<br><span class='metric'>{len(sentences)}</span></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='card'>Characters<br><span class='metric'>{len(text)}</span></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='card'>Unique Words<br><span class='metric'>{len(set(words))}</span></div>", unsafe_allow_html=True)

# ================= TOKENIZATION =================
if text.strip():
    st.header("📊 Tokenization Analysis")

    tokenizers = {
        "Sentence": sent_tokenize(text),
        "Word": word_tokenize(text),
        "Blank Line": blankline_tokenize(text),
        "Whitespace": WhitespaceTokenizer().tokenize(text),
        "WordPunct": WordPunctTokenizer().tokenize(text)
    }

    stats = []
    for name, toks in tokenizers.items():
        stats.append({
            "Tokenizer": name,
            "Total Tokens": len(toks),
            "Unique Tokens": len(set(toks))
        })

    st.dataframe(pd.DataFrame(stats))

# ================= NLU =================
if text.strip():
    st.header("🧠 Linguistic Understanding")

    filtered = [w.lower() for w in words if w.lower() not in stop_words and w.isalpha()]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Stemming")
        st.write([stemmer.stem(w) for w in filtered][:30])

    with col2:
        st.subheader("Lemmatization")
        st.write([lemmatizer.lemmatize(w) for w in filtered][:30])

    st.subheader("POS Distribution")
    pos_tags = nltk.pos_tag(filtered)
    pos_freq = Counter(tag for _, tag in pos_tags)
    st.bar_chart(pd.DataFrame(pos_freq.values(), index=pos_freq.keys()))

    st.subheader("Named Entity Recognition")
    doc = nlp(text)
    st.write([(ent.text, ent.label_) for ent in doc.ents])

# ================= FEATURE ENGINEERING =================
if text.strip():
    st.header("⚙️ Feature Engineering")

    corpus = []
    for s in sentences:
        clean = re.sub("[^a-zA-Z]", " ", s).lower().split()
        clean = [lemmatizer.lemmatize(w) for w in clean if w not in stop_words]
        corpus.append(" ".join(clean))

    cv = CountVectorizer(max_features=15)
    bow = cv.fit_transform(corpus).toarray()
    st.subheader("Bag of Words")
    st.dataframe(pd.DataFrame(bow, columns=cv.get_feature_names_out()))

    tf = TfidfVectorizer()
    tfidf = tf.fit_transform([text]).toarray()[0]
    st.subheader("TF-IDF")
    st.dataframe(
        pd.DataFrame({
            "Word": tf.get_feature_names_out(),
            "Score": tfidf
        }).sort_values("Score", ascending=False).head(10)
    )

    st.subheader("Word2Vec")
    model = Word2Vec([w.split() for w in corpus], vector_size=100, min_count=1)

    probe = st.text_input("Enter a word")
    if probe in model.wv:
        st.dataframe(
            pd.DataFrame(model.wv.most_similar(probe),
            columns=["Word", "Similarity"])
        )

# ================= WORDCLOUD =================
if text.strip():
    st.header("✨ Visual Insights")

    wc = WordCloud(
        width=900,
        height=400,
        background_color="black",
        colormap="cool"
    ).generate(text)

    fig, ax = plt.subplots(figsize=(10,4))
    ax.imshow(wc)
    ax.axis("off")
    st.pyplot(fig)

st.markdown("---")
st.caption("👩‍💻 Built by Akanksha Khurana")








