import pandas as pd
import numpy as np
import sklearn
import spacy
import nltk
import re
import torch
import statsmodels as sm
import seaborn as sns
import matplotlib.pyplot as plt
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

nltk.download('punkt')
nlp = spacy.load("en_core_web_sm")


def load_data_df(dir_path="data", file_name="corcerers_stone_harry_poter1", file_fmt="txt"):
    file_path = f"{dir_path}/{file_name}.{file_fmt}"
    return pd.read_csv(file_path)


def load_data(dir_path="data", file_name="corcerers_stone_harry_poter1", file_fmt="txt"):
    file_path = f"{dir_path}/{file_name}.{file_fmt}"
    with open(file_path, "r", encoding="utf-8") as f:
        book_text = f.read()
        return book_text


def preprocess_corpus(text):
    """
    Before tokenization, normalize the text:

        - Remove extra whitespace and line breaks.
        - Convert text to lowercase (for case insensitivity).
        - Remove non-alphanumeric characters (punctuation, special symbols).
        - Normalize quotes and dashes.
    :return: processed text
    """
    text = clean_text(text)
    sentences = nltk.sent_tokenize(text)
    passages = [" ".join(sentences[i:i + 5]) for i in range(0, len(sentences), 5)]
    passages = [lemmatize(p) for p in passages]
    return passages

    pass


def clean_text(text):
    """
    Before tokenization, normalize the text:

        - Remove extra whitespace and line breaks.
        - Convert text to lowercase (for case insensitivity).
        - Remove non-alphanumeric characters (punctuation, special symbols).
        - Normalize quotes and dashes.
    :param text: text to process
    :return: processed text
    """
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces and newlines
    text = re.sub(r'[“”‘’]', '"', text)  # Normalize quotes
    text = re.sub(r'[-—]', ' ', text)  # Normalize dashes
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove special characters
    return text


def lemmatize(text):
    """

    :param text:
    :return:
    """
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop])


def create_chunks(text):
    """

    :param text:
    :return:
    """
    sentences = nltk.sent_tokenize(text)
    passage_size = 5
    passages = [" ".join(sentences[i:i + passage_size]) for i in range(0, len(sentences), passage_size)]
    return passages


def preprocess_text(text):
    text = clean_text(text)
    passages = create_chunks(text)
    passages = [lemmatize(p) for p in passages]
    return passages


if __name__ == '__main__':
    book_text = load_data()
    # print(book_text)
    passages = preprocess_corpus(book_text)

    query = ["Who is Harry Potter?"]

    vectorizer = TfidfVectorizer(stop_words="english") #, max_df=0.9, min_df=2)
    tfidf_matrix = vectorizer.fit_transform(passages)
    query_vector = vectorizer.transform(query)

    similarity_scores = cosine_similarity(query_vector, tfidf_matrix)

    ranked_indices = np.argsort(similarity_scores[0])[::-1]
    print(ranked_indices)
    print("Top 5 Relevant Passages:")
    for i in range(5):

        idx = ranked_indices[i]
        print(f"\nRank {i + 1}, Score: {similarity_scores[0][idx]:.4f}")
        print(passages[idx])
