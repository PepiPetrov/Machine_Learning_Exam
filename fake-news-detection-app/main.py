import streamlit as st
from joblib import load

from custom_classes import *

normalizer = TextNormalizer()

fake_news_classifier = load("fake-news-detection-pipeline.model")
# fake_news_classifier = load("best-pipe.model")


st.write("# Fake News Detection App")

def check_news(title, text):
    normalized_title = normalizer.transform([title])
    label = fake_news_classifier.predict(normalized_title)[0]

    if label == 0:
        st.markdown("<h3 style=\"color:green\">News is real</h3>", unsafe_allow_html = True)
    elif label == 1:
        st.markdown("<h3 style=\"color:red\">News is fake</h3>", unsafe_allow_html = True)

title = st.text_input("Fake news title")

text = st.text_area("Fake news text")

btn = st.button("Check news", on_click = lambda: check_news(title, text))
