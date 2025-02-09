# Passage Retrieval - Home Assignment

This repository contains an implementation of **passage retrieval** for *Harry Potter and the Sorcerer's Stone* using **lexical (TF-IDF) and semantic (MiniLM, FAISS) retrieval methods**. The system allows users to input a query and retrieve the most relevant passages from the book.<br>

Search Engine App (Streamlit) - https://sleek-ha.streamlit.app/  

# Configuration

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm

```

The retrieval approaches can be tested by running:

```bash
python Sleek/lexical_retrieval.py
```
```bash
python Sleek/semantic_search.py
```

Evaluation pipeline can be found in notebook 'results_open_question.ipynb', or run:
```bash
python Sleek/evaluation.py
```



##### Bib

https://www.pinecone.io/learn/offline-evaluation/#Metrics-in-Information-Retrieval
https://faiss.ai/cpp_api/struct/structfaiss_1_1IndexIVF.html