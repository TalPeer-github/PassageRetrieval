import pandas as pd
import numpy as np
from Sleek.utils import *


def compare(k=5):
    tf_idf_results = pd.read_csv('Sleek/data/idx_passages_retrieved_TFIDF_k10.csv').drop(columns=['Unnamed: 0'])
    semantic_search = pd.read_csv('Sleek/data/idx_passages_retrieved_IVF_nprob2.csv').drop(columns=['Unnamed: 0'])

    for k in range(1, k + 1):
        recall_scores = []
        mrr_scores = []
        ndcg_scores = []
        chapters_mrr_scores = []
        chapters_recall_scores = []

        for i, query in enumerate(queries):
            actual = semantic_search.iloc[i].tolist()
            predicted = tf_idf_results.iloc[i].tolist()

            recall = recall_k(actual=actual, predicted=predicted, k=k)
            recall_scores.append(recall)

            mrr = mrr_k(actual=actual, predicted=predicted, k=k)
            mrr_scores.append(mrr)

            chapt_mrr = chapter_mrr(actual, predicted, k=k)
            chapters_mrr_scores.append(chapt_mrr)

            chapt_recall = chapter_recall(actual, predicted, k=k)
            chapters_recall_scores.append(chapt_recall)

            ndcg = ndcg_k(actual=actual, predicted=predicted, k=k)
            ndcg_scores.append(ndcg)

        avg_recall = np.mean(recall_scores)
        avg_mrr = np.mean(mrr_scores)
        avg_ndcg = np.mean(ndcg_scores)
        avg_crecall = np.mean(chapters_recall_scores)
        avg_cmrr = np.mean(chapters_mrr_scores)
        print("================================================")
        print(f"Max Recall@{k}: {np.max(recall_scores):.3f} (Query {np.argmax(recall_scores) + 1})")
        print(f"Max MRR@{k}: {np.max(mrr_scores):.3f} (Query {np.argmax(mrr_scores) + 1})")
        print(f"Max NDCG@{k}: {np.max(ndcg_scores):.3f} (Query {np.argmax(ndcg_scores) + 1})")
        print(f"Max C-Recall@{k}: {np.max(chapters_recall_scores):.3f} (Query {np.argmax(chapters_recall_scores) + 1})")
        print(f"Max CMRR@{k}: {np.max(chapters_mrr_scores):.3f} (Query {np.argmax(chapters_mrr_scores) + 1})\n")

        print(f"Avg Recall@{k}: {avg_recall:.3f}")
        print(f"Avg MRR@{k}: {avg_mrr:.3f}")
        print(f"Avg NDCG@{k}: {avg_ndcg:.3f}")
        print(f"Avg C-Recall@{k}: {avg_crecall:.3f}")
        print(f"Avg CMRR@{k}: {avg_cmrr:.3f}")
        print("================================================")


def recall_k(actual, predicted, k):
    """
     By increasing K to N or near N, we can return a perfect score every time, so relying solely on recall@K can be deceptive.
     Furthermore, it is an order-unaware metric.
    :param actual: Semantic search as "ground truth"
    :param predicted: Lexical retrieve with TF-IDF
    :param k: k relevance
    :return: recall@k scores
    """
    act_set = set(actual)
    pred_set = set(predicted[:k])
    result = round(len(act_set & pred_set) / float(len(act_set)), 4)
    return result


def mrr_k(actual, predicted, k):
    """
    Mean Reciprocal Rank (MRR) measures how early the first relevant result appears.
    :param actual: Semantic search as "ground truth"
    :param predicted: Lexical retrieve with TF-IDF
    :param k: Top-K results to consider
    :return: MRR@K score
    """
    cumm_r = 0
    for act in actual:
        for rank, item in enumerate(predicted[:k], start=1):
            if item == act:
                cumm_r += (1 / rank)
    return cumm_r


def chapter_mrr(actual, predicted, k):
    """
    MRR@k score, adjusted for accounting Chapters instead of passages
    (By checking if the passages returned belongs to the same chapter)
    :param actual: Semantic search as "ground truth"
    :param predicted: Lexical retrieve with TF-IDF
    :param k: Top-K results to consider
    :return: Chapter_MRR@K score
    """
    actual_chapters = passages_df.iloc[actual]['str_idx'].tolist()
    pred_chapters = passages_df.iloc[predicted]['str_idx'][:k].tolist()
    cumm_mrr = 0
    for rank, chapter in enumerate(pred_chapters[:k], 1):
        if chapter in actual_chapters:
            cumm_mrr += (1 / rank)
    return cumm_mrr


def chapter_recall(actual, predicted, k):
    """
    Recall@k score, adjusted for accounting Chapters instead of passages.
    (By checking if the passages returned belongs to the same chapter)
    :param actual: Semantic search as "ground truth"
    :param predicted: Lexical retrieve with TF-IDF
    :param k: Top-K results to consider
    :return: Chapter_Recall@K score
    """
    actual_chapters = set(passages_df.iloc[actual]['str_idx'].tolist())
    pred_chapters = set(passages_df.iloc[predicted]['str_idx'][:k].tolist())
    result = round(len(actual_chapters & pred_chapters) / float(len(actual_chapters)), 4)
    return result


def ndcg_k(actual, predicted, k):
    """
    Normalized Discounted Cumulative Gain (NDCG) for ranking quality measure.
    :param actual: Semantic search as "ground truth"
    :param predicted: Lexical retrieve with TF-IDF
    :param k: Top-K results to consider
    :return: NDCG@K score
    """

    def dcg(scores):
        return sum((rel / np.log2(idx + 2)) for idx, rel in enumerate(scores))

    relevance_scores = [1 if item in actual else 0 for item in predicted[:k]]
    dcg_k = dcg(relevance_scores)
    idcg_k = dcg(sorted(relevance_scores, reverse=True))
    result = round(dcg_k / idcg_k, 4) if idcg_k > 0 else 0.0
    return result


passages_df = pd.read_csv('data/clean_chunks.csv')
compare()
