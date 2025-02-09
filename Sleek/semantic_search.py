import pandas as pd
import numpy as np
import spacy
import re
import faiss
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle
from utils import queries, complex_queries

EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
nlp = spacy.load("en_core_web_sm")


def embedding_model():
    model = SentenceTransformer(EMBEDDING_MODEL)
    return model


def embedd_dataset(dataset, model: SentenceTransformer = SentenceTransformer('all-MiniLM-L6-v2'),
                   text_field: str = 'content') -> np.ndarray:
    """
    Load a dataset and embedd the text field using a sentence-transformer model
    :param dataset: pandas.DataFrame object to load
    :param model: The model to use for embedding
    :param text_field: The field in the dataset that contains the text
    :return: np.ndarray: A tuple containing the dataset and the embeddings
    """

    embeddings = model.encode(dataset[text_field], show_progress_bar=True)
    return embeddings


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


def build_faiss_lsh_index(embeddings: np.ndarray, dim: int, nbits: int, ):
    """
    This function builds a Faiss LSH index.
    Args:
        embeddings: An array of shape (n_index, dim) containing the index vectors.
        dim: The dimensionality of the vectors.
        nbits: The number of bits to use in the hash.
    Returns:
        A Faiss LSH index.
    """
    index = faiss.IndexLSH(dim, nbits)
    index.add(embeddings)
    return index


def train_index(index, sentence_embeddings: np.ndarray):
    index.add(sentence_embeddings)


def encode_query(query: str, model):
    return model.encode(query)


def compute_recall_at_k(nn_gt: np.ndarray, ann: np.ndarray, k: int,
                        ):
    """
    This function computes the recall@k.
    Args:
        nn_gt: The ground truth nearest neighbors, which are the closest relatives to chunks.
        ann: The approximate nearest neighbors.
        k: The number of nearest neighbors to consider.
    Returns:
        The recall@k.
    """
    return round(sum([len(set(ann[i]) & set(nn_gt[i])) / k for i in range(len(ann))]) / len(ann), 3)


def load_df(dir_path: str, file_name: str, file_fmt: str):
    file_path = f"{dir_path}/{file_name}.{file_fmt}"
    return pd.read_csv(file_path)



def create_splits(texts, split_chunk_size, split_overlap):
    """
    :param texts: texts to create splits for
    :param split_chunk_size: The maximum size of a chunk, where size is determined by the length_function.
    :param split_overlap:Target overlap between chunks. Overlapping chunks helps to mitigate loss of information when context is divided between chunks.
    :return: chunked texts
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=split_chunk_size,
        chunk_overlap=split_overlap
    )

    texts_chunks = text_splitter.split_text(texts)

    print(f'Created {len(texts_chunks)} chunks.')
    return texts_chunks


def create_book_chunks(book_df, split_chunk_size=1000, split_overlap=0.3):
    chunks = []
    for i, chapter_content in enumerate(book_df['processed_content']):
        chapter_idx = book_df['str_idx'].iloc[i]
        chapter_chunks = create_splits(chapter_content, split_chunk_size=split_chunk_size, split_overlap=split_overlap)
        for j, c_chunk in enumerate(chapter_chunks):
            chunks.append((chapter_idx, j, c_chunk))
    try:
        chunks_df = pd.DataFrame.from_records(chunks, columns=["str_idx", "chunk_id", "chunk"])
        chunks_df['processed_chunk'] = chunks_df['chunk'].apply(lambda chunk: clean_text(chunk))
        chunks_df.to_csv('data/clean_chunks.csv', index=False)
        return chunks_df
    except:
        print("chunks CSV file could not be created.")



def build_ivf_index(embeddings, dim: tuple, _nprob=None):
    nlist = 17
    quantizer = faiss.IndexFlatL2(dim[1])
    index = faiss.IndexIVFFlat(quantizer, dim[1], nlist)
    index.train(embeddings)
    index.add(embeddings)
    if _nprob is not None:
        index.nprobe = _nprob
    return index


def retrieve_top_passages(query, model, index, chunks, top_n=5):
    query_embedding = model.encode([clean_text(query)], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_n)
    chunk = chunks['chunk'].iloc[indices[0]].tolist()
    p_chapter = chunks['str_idx'].iloc[indices[0]].tolist()
    return chunk, p_chapter, indices[0]


def extract_entities(indices, chunks):
    passages = chunks.iloc[indices[0], :]
    for passage in passages:
        p_chapter = passage['str_idx']
        p_title = passage['title']
        p_content = passage['chunk']
        p_persons = passage['keywords']
    # doc = nlp(p_chapter)
    # persons_list = [entity.text for entity in doc.ents if entity.label_ == 'PERSON']
    return passages, p_chapter, p_title, p_content, p_persons


def retrieve(model, index, chunks):
    test_queries = queries + complex_queries
    retrieval_results = {query: [] for query in test_queries}
    for query in test_queries:
        r_chunks, r_chapters, indices = retrieve_top_passages(query, model, index, chunks)
        retrieval_results[query] = indices
    return retrieval_results


def save_embeddings(embeddings: np.ndarray, file_path="data/embeddings.pkl"):
    with open(file_path, "wb") as f:
        pickle.dump(embeddings, f)


def load_embeddings(file_path="data/embeddings.pkl"):
    with open(file_path, "rb") as f:
        loaded_embeddings = pickle.load(f)
    return loaded_embeddings


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
    doc = nlp(text)
    text = " ".join(token.lemma_ for token in doc)
    return text


def main(create_chunks_df=False):
    book_df = load_df(dir_path="data", file_name="book_df", file_fmt="csv")
    if create_chunks_df:
        chunks_df = create_book_chunks(book_df, split_chunk_size=1000, split_overlap=0.3)
    else:
        chunks_df = load_df(dir_path="data", file_name="chunks", file_fmt="csv")

    model = embedding_model()
    embeddings = embedd_dataset(
        dataset=chunks_df,
        model=model,
        text_field='processed_chunk',
    )
    save_embeddings(embeddings, file_path='data/clean_embeddings.pkl')


def start_search():
    chunks_df = pd.read_csv('data/clean_chunks.csv')
    embeddings = load_embeddings(file_path='data/clean_embeddings.pkl')
    embeddings_shape = embeddings.shape
    model = embedding_model()
    index = build_ivf_index(embeddings, embeddings_shape, _nprob=2)
    passages_retrieved_indices = retrieve(model, index, chunks_df)
    pd.DataFrame.from_dict(passages_retrieved_indices, orient="index").to_csv('data/idx_passages_retrieved_IVF_nprob2.csv')


if __name__ == "__main__":
    start_search()
