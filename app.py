import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import re
from scripts.youtube_comments import get_comments

from utils.text_cleaner import clean_text
def load_css():
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ---------------- PAGE CONFIG ----------------

st.set_page_config(page_title="AI Sentiment Intelligence",layout="wide")

# ---------------- LOAD CSS ----------------

def load_css():
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>",unsafe_allow_html=True)

load_css()

# ---------------- TITLE ----------------

st.markdown('<p class="main-title">AI Sentiment Intelligence</p>',unsafe_allow_html=True)
st.markdown('<p class="subtitle">Amazon Reviews • YouTube Comments Analysis</p>',unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------

model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# ---------------- PRODUCT IMAGE ----------------

def get_product_image(query):
    return f"https://source.unsplash.com/800x400/?{query}"

def extract_product(review):
    words = review.split()
    return " ".join(words[:3])

# ---------------- YOUTUBE VIDEO ID ----------------

def extract_video_id(url):
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern,url)
    return match.group(1) if match else None

# ---------------- PLATFORM SELECT ----------------

platform = st.selectbox(
"Choose Platform",
["Amazon Review Predictor","Amazon CSV Analysis","YouTube Comment Analysis"]
)

# =================================================
# AMAZON MANUAL REVIEW
# =================================================

if platform == "Amazon Review Predictor":

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("🛒 Amazon Product Review Predictor")

    review = st.text_area("Enter Amazon Product Review")

    if st.button("Analyze Review"):

        clean = clean_text(review)

        X = vectorizer.transform([clean])

        prediction = model.predict(X)[0]

        if prediction == 1:
            sentiment="Positive"
            st.success("Positive Review 😊")
        else:
            sentiment="Negative"
            st.error("Negative Review 😡")

        fig = px.pie(
        values=[1,0] if sentiment=="Positive" else [0,1],
        names=["Positive","Negative"],
        title="Sentiment Distribution"
        )

        st.plotly_chart(fig)

    st.markdown('</div>', unsafe_allow_html=True)

# =================================================
# AMAZON CSV ANALYSIS
# =================================================

if platform == "Amazon CSV Analysis":

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("📊 Amazon CSV Bulk Review Analysis")

    file = st.file_uploader("Upload Amazon Reviews CSV",type=["csv"])

    if file:

        df = pd.read_csv(file)

        review_col = df.columns[0]

        if st.button("Run Analysis"):

            df["clean"] = df[review_col].astype(str).apply(clean_text)

            X = vectorizer.transform(df["clean"])

            df["sentiment"] = model.predict(X)

            pos = sum(df["sentiment"])
            neg = len(df)-pos

            col1,col2 = st.columns(2)

            col1.metric("Positive Reviews",pos)
            col2.metric("Negative Reviews",neg)

            fig = px.pie(
            values=[pos,neg],
            names=["Positive","Negative"],
            title="Sentiment Distribution"
            )

            st.plotly_chart(fig)

            st.dataframe(df[[review_col,"sentiment"]].head())

    st.markdown('</div>', unsafe_allow_html=True)
# =================================================
# YOUTUBE ANALYSIS
# =================================================

if platform == "YouTube Comment Analysis":

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("▶️ YouTube Comment Sentiment Analysis")

    url = st.text_input("Paste YouTube Video URL")

    if st.button("Analyze YouTube Comments"):

        comments = get_comments(url)

        df = pd.DataFrame(comments, columns=["comment"])

        df["clean"] = df["comment"].apply(clean_text)

        X = vectorizer.transform(df["clean"])

        df["sentiment"] = model.predict(X)

        pos = sum(df["sentiment"])
        neg = len(df) - pos

        col1, col2 = st.columns(2)

        col1.metric("Positive Comments", pos)
        col2.metric("Negative Comments", neg)

        fig = px.pie(
            values=[pos, neg],
            names=["Positive", "Negative"],
            title="YouTube Sentiment Analysis"
        )

        st.plotly_chart(fig)

        st.dataframe(df.head())

    st.markdown('</div>', unsafe_allow_html=True)