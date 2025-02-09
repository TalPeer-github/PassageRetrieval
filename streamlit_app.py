import pickle
import re

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
    page_icon="📖",
    initial_sidebar_state="expanded"
)

st.sidebar.title("Sleek Home Assignment - Tal Peer")
st.sidebar.markdown("<h2 style='color: #374785;'>📘 Methodology</h2>", unsafe_allow_html=True)

st.sidebar.markdown("""
### Objectives
1. **Text Chunking**: The book is split into **518** chunks for passage retrieval, using RecursiveCharacterTextSplitter (langchain.text_splitter).
2. **Embedding Model**: The chosen **MiniLM* is 'all-MiniLM-L6-v2', available in Hugging Face models platform for Embeddings creation (num_chunks, 384).
3. **FAISS Indexing**: These vectors are stored in a **FAISS Index** for fast similarity search. Both L2 Index & Inverted File Index were tested, where IVF index was chosen as the final index based on better performance. 

""", unsafe_allow_html=True)

st.markdown("""
    <style>
    /* Background & Text */
    body { background-color: #F5F7FA; color: #1E2A38; }
    .stApp { background-color: #F5F7FA; }
    .stTextInput>div>div>input {
        background-color: white;
        border-radius: 10px;
        border: 1px solid #DCE1E9;
        padding: 8px;
        box-shadow: none;
    }
    .stTextInput>div>div>input:focus {
        border: 1px solid #3979F0;
        box-shadow: 0px 0px 5px rgba(57, 121, 240, 0.5);
    }
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #3979F0, #1D4EDD);
        color: white;
        border-radius: 10px;
        padding: 8px 15px;
        border: none;
        transition: 0.3s ease-in-out;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #1D4EDD, #3979F0);
    }
    /* Search Results */
    .result-card {
        background-color: white;
        padding: 10px;
        margin-bottom: 10px;
        border-radius: 10px;
        box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.1);
    }
    .highlight {
        background-color: #DCE1E9;
        padding: 2px 5px;
        border-radius: 4px;
    }
    </style>
""", unsafe_allow_html=True)

# Title & Header
st.markdown("<h1 style='text-align: center; color: #374785;'>📖 Harry Potter Search Engine</h1>",
            unsafe_allow_html=True)
# st.markdown(
#     "<h3 style='text-align: center; color: #1E2A38;'>Find relevant passages from *Harry Potter and the Sorcerer's Stone*</h3>",
#     unsafe_allow_html=True)

alt.themes.enable("dark")
st.markdown("<h4 style='text-align: center; color: #374785;'>Passage Retrieval TF-IDF & Semantic Search with MiniLM</h4>",
            unsafe_allow_html=True)

queries = ["What is the Sorcerer’s Stone?",
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

# def retrieve_top_passages(query, model, index, chunks, top_n=5):
#     query_embedding = model.encode([clean_text(query)], convert_to_numpy=True)
#     distances, indices = index.search(query_embedding, top_n)
#     retrieved_passages = chunks['chunk'].iloc[indices[0]].tolist()
#     return retrieved_passages


def clean_text(text):
    """
    Cleans text by:
    - Normalizing Unicode characters
    - Removing special characters and extra whitespace
    - Lowercasing text
    - Keeping only alphanumeric characters and essential punctuation
    """
    text = re.sub(r"[^a-zA-Z0-9\s.,?!]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.lower()
    return text

def retrieve(model, index, chunks):
    test_queries = queries + complex_queries
    retrieval_results = {query: [] for query in test_queries}
    for query in test_queries:
        retrieval_results[query] = retrieve_top_passages(query, model, index, chunks)
    return retrieval_results

def build_IVFIndex(embeddings, dim: tuple,n_list=17, n_prob = None):
    quantizer = faiss.IndexFlatL2(dim[1])
    index = faiss.IndexIVFFlat(quantizer, dim[1], n_list)
    index.train(embeddings)
    index.add(embeddings)
    if n_prob is not None:
        index.nprobe = n_prob
    return index




def load_embeddings(file_path="data/embeddings.pkl"):
    with open(file_path, "rb") as f:
        loaded_embeddings = pickle.load(f)
    return loaded_embeddings


def build_faiss_index(embeddings: np.ndarray, dim: tuple):
    index = faiss.IndexFlatL2(dim[1])
    index.add(embeddings)
    return index


EMBEDDING_MODEL = 'all-MiniLM-L6-v2'


def embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)

def retrieve_top_passages(query, model, index, chunks, top_n=5):
    query_embedding = model.encode([clean_text(query)], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_n)
    chunk = chunks['chunk'].iloc[indices[0]].tolist()
    p_chapter = chunks['str_idx'].iloc[indices[0]].tolist()
    distances = distances[0]
    return zip(chunk,p_chapter, distances)


def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9\s.,?!]", "", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


model = embedding_model()
chunks_df = pd.read_csv('Sleek/data/clean_chunks.csv')
embeddings = load_embeddings('Sleek/data/clean_embeddings.pkl')
dim = embeddings.shape

st.markdown("<h4 style='color: #374785;'>Choose Index Type:</h4>", unsafe_allow_html=True)
index_name = st.radio("Available Indices:",options=["L2 Flat Indes", "IVF Index"], horizontal = True)
ivf_selected = index_name == "IVF Index"

if st.button("Set Index"):
    if ivf_selected:
        choose_n_cells = st.number_input("Choose number of Vernoi Cells", placeholder="deafult = 17", value=17, min_value=17,max_value=50)
        choose_n_prob = st.number_input("Choose number of n_prob", placeholder="deafult = 2", value=2, min_value=1,max_value=8)
        n_prob = 2 if choose_n_prob == "" else int(choose_n_prob)
        n_cells = 17 if choose_n_cells == "" else int(choose_n_cells)
        if st.button("Set Chosen Params"):
            index = build_IVFIndex(embeddings, dim,n_cells, n_prob=n_prob)
    else:
        index = build_faiss_index(embeddings, dim)
else:
    index = build_faiss_index(embeddings, dim)

st.markdown("<h4 style='color: #374785;'>Try One of the Test Queries:</h4>", unsafe_allow_html=True)
selected_query = st.selectbox("Choose a query:", queries + complex_queries)

if st.button("Search with Selected Query"):
    retrieved_passages = retrieve_top_passages(selected_query, model, index, chunks_df)
    st.markdown("<h4 style='color: #021526;'>Relevant Passages</h4>", unsafe_allow_html=True)

    for i, (passage,chapter,dist) in enumerate(retrieved_passages):
        with st.container(height=300):
            st.markdown(f"<h6 style='color: #374785;'>Result {i+1} - {chapter}</h6>", unsafe_allow_html=True)
            st.markdown(f"{passage}",unsafe_allow_html=True)


st.markdown("<h4 style='color: #374785;'>Search with Your Own Query</h4>", unsafe_allow_html=True)
query = st.text_input("Enter your query:", placeholder="e.g., What does the Mirror of Erised show Harry?")
if query == "":
    query = "What does the Mirror of Erised show Harry?"
if st.button("Search Specific Query"):
    retrieved_passages = retrieve_top_passages(query, model, index, chunks_df)
    st.markdown("<h4 style='color: #1D4EDD;'>Retrieved Passages</h4>", unsafe_allow_html=True)
    for i, (passage,chapter,dist) in enumerate(retrieved_passages):
        with st.container(height=400):
            st.markdown(f"<h6 style='color: #374785;'>Result {i+1} - {chapter}</h6>", unsafe_allow_html=True)
            st.markdown(f"{passage}",unsafe_allow_html=True)



