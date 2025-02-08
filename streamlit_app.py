import pickle

import numpy as np
import streamlit as st
import pandas as pd
import altair as alt
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import faiss
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="Passage Retrieval - Sleek Home Assignment",
    page_icon="🏂",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")
st.title("Sleek Home Assignment - Tal Peer")
st.header("Explore Passage Retrieval and Semantic Search with MiniLM")

queries = ["What is the Sorcerer’s Stone and what does it do?",
           "How does Harry get into Gryffindor?",
           "What does the Mirror of Erised show Harry?",
           "Who helps Harry get past Fluffy?",
           "What happens during Harry’s first Quidditch match?", ]
complex_queries = [
    "How do Harry, Ron, and Hermione manage to bypass each of the obstacles guarding the Sorcerer’s Stone?",
    "what role does Fluffy play in the protection system?",
    "Snape’s and Quirrell’s interference Harrys first quidditch match"]


def load_embeddings(file_path="data/embeddings.pkl"):
    with open(file_path, "rb") as f:
        loaded_embeddings = pickle.load(f)
    return loaded_embeddings


def build_faiss_flatl2_index(embeddings: np.ndarray, dim: tuple):
    """
    This function builds a Faiss flat L2 index.
    Args:
        embeddings: An array of shape (n_index, dim) containing the index vectors.
        dim: The dimensionality of the vectors.
    Returns:
        A Faiss flat L2 index.
    """
    index = faiss.IndexFlatL2(dim[1])
    index.add(embeddings)
    return index


EMBEDDING_MODEL = 'all-MiniLM-L6-v2'


def embedding_model():
    model = SentenceTransformer(EMBEDDING_MODEL)
    return model


model = embedding_model()
chunks_df = pd.read_csv('Sleek/data/clean_chunks.csv')
embeddings = load_embeddings('Sleek/data/clean_embeddings.pkl')
dim = embeddings.shape
index = build_faiss_flatl2_index(embeddings, dim)

st.header('Tested Queries')
st.write('Please select a query from the list below:')
selected_position = st.selectbox('Select Query', queries + complex_queries)

colors = ['#164863', '#427D9D', '#9BBEC8', '#DDF2FD', '#F2EBE9', '#6B818C']

st.title("📖 Harry Potter Search Engine")
st.markdown("Search *Harry Potter and the Sorcerer's Stone* for relevant passages.")

query = st.text_input("Enter your query:")
if query:
    query_embedding = model.encode(query).reshape(1, -1)
    _, indices = index.search(query_embedding, 5)

    st.subheader("🔍 Relevant Passages")
    for i, idx in enumerate(indices[0]):
        st.markdown(f"**{i + 1}.** {chunks_df[idx]}")
