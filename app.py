import streamlit as st
import nltk, spacy, re
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

# ================= SETUP =================
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

try:
    nlp = spacy.load("en_core_web_sm")
except:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# ================= UI =================
st.set_page_config("NLP Intelligence Studio", layout="wide")

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

st.title("🧠 NLP Intelligence Studio — **ENGINEER LEVEL**")
st.caption("NLU • Linguistic Analysis • Feature Engineering • Semantic Modeling • Explainability")

text = st.text_area("✍️ Paste Paragraph", height=220)

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# ======================================================
# TEXT STATISTICS DASHBOARD
# ======================================================
if text.strip():
    words = word_tokenize(text)
    sentences = sent_tokenize(text)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='card'>Total Words<br><span class='metric'>{len(words)}</span></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='card'>Sentences<br><span class='metric'>{len(sentences)}</span></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='card'>Characters<br><span class='metric'>{len(text)}</span></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='card'>Unique Words<br><span class='metric'>{len(set(words))}</span></div>", unsafe_allow_html=True)

# ======================================================
# TOKENIZATION — COMPARATIVE ANALYSIS (PRO)
# ======================================================
if text.strip():
    st.header("📊 Tokenization Analysis (Comparative NLU)")

    tokenizers = {
        "Sentence": sent_tokenize(text),
        "Word": word_tokenize(text),
        "Blank Line": blankline_tokenize(text),
        "Whitespace": WhitespaceTokenizer().tokenize(text),
        "WordPunct": WordPunctTokenizer().tokenize(text)
    }

    stats = []
    for name, toks in tokenizers.items():
        if len(toks) > 0:
            stats.append({
                "Tokenizer": name,
                "Total Tokens": len(toks),
                "Unique Tokens": len(set(toks)),
                "Avg Token Length": round(sum(len(t) for t in toks)/len(toks),2)
            })

    st.dataframe(pd.DataFrame(stats))

    st.markdown("""
**Insight:**  
Tokenization strategy directly impacts linguistic structure and downstream models.  
WordPunct captures punctuation semantics, while Whitespace is computationally cheap but noisy.
""")

# ======================================================
# NLU — LINGUISTIC UNDERSTANDING
# ======================================================
if text.strip():
    st.header("🧠 Natural Language Understanding (NLU)")

    filtered = [w.lower() for w in words if w.lower() not in stop_words and w.isalpha()]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Stemming")
        st.write([stemmer.stem(w) for w in filtered][:40])

    with col2:
        st.subheader("Lemmatization")
        st.write([lemmatizer.lemmatize(w) for w in filtered][:40])

    st.subheader("POS Tag Distribution")
    pos_tags = nltk.pos_tag(filtered)
    pos_freq = Counter(tag for _, tag in pos_tags)
    st.bar_chart(pd.DataFrame(pos_freq.values(), index=pos_freq.keys()))

    st.subheader("Named Entity Recognition (NER)")
    doc = nlp(text)
    st.write([(ent.text, ent.label_) for ent in doc.ents])

# ======================================================
# NLP ALGORITHMS — FEATURE ENGINEERING
# ======================================================
if text.strip():
    st.header("⚙️ NLP Algorithms (Feature Engineering)")

    corpus = []
    for s in sentences:
        review = re.sub('[^a-zA-Z]', ' ', s).lower().split()
        review = [lemmatizer.lemmatize(w) for w in review if w not in stop_words]
        corpus.append(' '.join(review))

    st.subheader("Bag of Words (BoW)")
    cv = CountVectorizer(max_features=15)
    bow = cv.fit_transform(corpus).toarray()
    st.dataframe(pd.DataFrame(bow, columns=cv.get_feature_names_out()))

    st.subheader("TF-IDF Feature Importance")
    tf = TfidfVectorizer()
    tfidf = tf.fit_transform([text]).toarray()[0]
    df_imp = pd.DataFrame({
        "Word": tf.get_feature_names_out(),
        "Score": tfidf
    }).sort_values("Score", ascending=False).head(10)
    st.dataframe(df_imp)

    st.subheader("Word2Vec — Semantic Space")
    w2v_model = Word2Vec([w.split() for w in corpus], vector_size=100, window=5, min_count=1)

    probe = st.text_input("Semantic probe word (e.g. intelligence, learning)")
    if probe and probe in w2v_model.wv:
        df_sim = pd.DataFrame(w2v_model.wv.most_similar(probe),
                              columns=["Word", "Similarity"])
        st.dataframe(df_sim)

# ======================================================
# NLG + VISUALIZATION
# ======================================================
if text.strip():
    st.header("✨ NLG & Visual Intelligence")

    wc = WordCloud(
        width=900,
        height=400,
        background_color="#020617",
        colormap="cool"
    ).generate(text)

    plt.figure(figsize=(10,4))
    plt.imshow(wc)
    plt.axis("off")
    st.pyplot(plt)

    st.subheader("Auto Insight Generation (NLG)")
    top_terms = [w for w, _ in Counter(filtered).most_common(6)]
    st.write(
        "This document semantically focuses on "
        + ", ".join(top_terms)
        + ", indicating dominant conceptual themes in the text."
    )
