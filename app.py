import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd

# ---------- NLTK SETUP ----------
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="NLP Intelligence Studio",
    page_icon="🧠",
    layout="wide"
)

# ---------- STYLING ----------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg,#020617,#020617);
}
.big-title {
    font-size:48px;
    font-weight:800;
    background: linear-gradient(90deg,#22d3ee,#38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.card {
    background:#020617;
    padding:20px;
    border-radius:15px;
    box-shadow:0 0 20px rgba(56,189,248,0.2);
}
</style>
""", unsafe_allow_html=True)

# ---------- TITLE ----------
st.markdown('<div class="big-title">🧠 NLP Intelligence Studio</div>', unsafe_allow_html=True)
st.caption("Advanced Natural Language Processing Playground")

# ---------- INPUT ----------
text = st.text_area(
    "✍️ Enter your text",
    height=200,
    placeholder="Paste or type any English text here..."
)

if text.strip():

    st.markdown("---")

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

    # ---------- WORD FREQUENCY ----------
    freq = nltk.FreqDist(lemmas)
    df = pd.DataFrame(freq.most_common(10), columns=["Word", "Frequency"])

    st.subheader("🔹 Top 10 Word Frequency")
    st.dataframe(df, use_container_width=True)

    # ---------- WORDCLOUD ----------
    if len(text) < 5000:
        st.subheader("🔹 Word Cloud")
        wc = WordCloud(
            width=900,
            height=400,
            background_color="#020617",
            colormap="cool"
        ).generate(" ".join(lemmas))

        fig, ax = plt.subplots()
        ax.imshow(wc)
        ax.axis("off")
        st.pyplot(fig)

else:
    st.info("👆 Enter some text to activate NLP analysis")

# ---------- FOOTER ----------
st.markdown("---")
st.caption("👩‍💻 Built by **Akanksha Khurana** | Streamlit NLP Project")



