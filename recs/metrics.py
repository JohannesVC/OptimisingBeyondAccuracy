"""
Metrics Module
---------------
This module provides a comprehensive suite of evaluation metrics tailored for recommender systems. 
It includes rank-aware metrics (e.g., MRR, DCG, NDCG) to assess recommendation performance from multiple angles. In addition, normative metrics such as diversity, novelty, and prediction coverage are available to evaluate recommendation quality at the system level.

Key Features:
    - Ranking Metrics: Functions to compute metrics such as AUC, MRR (mean reciprocal rank), DCG, and NDCG.
    - Novelty Metrics: Functions to compute surprisal and expected popularity complement (EPC) to evaluate the unexpectedness of recommendations.
    - Diversity Metrics: Functions to compute intra-list diversity using both slow and fast implementations.
    - Coverage Metrics: Functions to compute prediction coverage and catalog coverage.
    - Aggregated Metrics: Functions to compute multiple metrics at once, including both standard and normative metrics.

Adaptations:
    - Portions of the implementation are adapted from open-source projects including:
        • recmetrics (statisticianinstilettos) [1]
        • mrec (Mendeley) [2]
        • Microsoft Recommenders [3]

Usage:
    Import this module and utilize the provided functions to compute various metrics based on input predictions and ground truth data,
    which can be provided as NumPy arrays or SciPy sparse matrices.

Sources:
    [1] https://github.com/statisticianinstilettos/recmetrics/blob/master/recmetrics/metrics.py
    [2] https://github.com/Mendeley/mrec/blob/master/mrec/evaluation/metrics.py
    [3] https://microsoft-recommenders.readthedocs.io/en/latest/
"""
from collections import Counter
import random
from tqdm.notebook import tqdm
from typing import List
import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import (f1_score, log_loss, mean_squared_error, 
                            #  ndcg_score, dcg_score, 
                             roc_auc_score, accuracy_score)
                   
def mrr_score_(y_true, y_score):
    """
    Compute a MRR score metric. 
    This is a variant to the standard MRR score,
    used in recommender package (e.g. for MIND leaderboard).

    Args:
        y_true (np.ndarray): Ground-truth labels.
        y_score (np.ndarray): Predicted labels.

    Returns:
        numpy.ndarray: mrr scores.
    """
    # Return indices of sorted array (reversed) and reorder y_true with them
    order = np.argsort(y_score)[::-1]   
    y_true = np.take(y_true, order)

    # compute the reciprocal rank score for each element
    rr_score = y_true / (np.arange(y_true.shape[0]) + 1) 

    total = np.sum(y_true)
    if total == 0:
        return 0.0
    # normalise the score by the sum of true labels
    return np.sum(rr_score) / total


def dcg_score(y_true, y_score, k=10):
    """Computing dcg score metric at k.

    Args:
        y_true (np.ndarray): Ground-truth labels.
        y_score (np.ndarray): Predicted labels.

    Returns:
        np.ndarray: dcg scores.
    """
    k = min(np.shape(y_true)[-1], k)
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(y_true.shape[0]) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    """Computing ndcg score metric at k.

    Args:
        y_true (np.ndarray): Ground-truth labels.
        y_score (np.ndarray): Predicted labels.

    Returns:
        numpy.ndarray: ndcg scores.
    """
    best = dcg_score(y_true, y_true, k)
    if best == 0:
        return 0.0
    
    actual = dcg_score(y_true, y_score, k)
    return actual / best

def custom_auc(y_true, y_pred):
    """
    Computes AUC ignoring padding altogether. 
    Every zero in the zero-padding counts towards the threshold.
    """
    # Mask out padding and flatten
    # Note that this does it by `y_pred`, which is a score
    mask = y_pred != 0.0
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    return roc_auc_score(y_true, y_pred)

# -------- Novelty, diversity, and coverage ------------

def slow_intra_list_diversity(y_pred, imprs, embedding_matrix, topk, alpha=0.15):
    """
    Calculate the diversity of recommendations for each user.
   
    Args:        
        y_pred (np.array): Array of predicted item interactions (recommendations) of shape num behaviors x num impressions.
        imprs (np.array): Array from dataloader.init_behaviors() 
        embedding_matrix (np.array): Matrix of item interaction embeddings.
        y_true (np.array): Array of true item interactions.

    Returns:
        np.array: Array of diversity scores for each user.
    """
    sessions = np.shape(y_pred)[0] # number of item interactions
    diversity_scores = np.zeros(sessions)

    for i, preds in tqdm(enumerate(y_pred), 
        total=sessions, desc="intra_list_diversity"):   

        # reorder the imprs with `order`
        order = np.argsort(preds)[::-1][:topk] # rank high to low        
        session_impr = np.take(imprs, order)         
        
        # Get feature vectors for the predicted items
        predicted_item_features = embedding_matrix[session_impr] # array of recommended news embeddings
        
        # Calculate pairwise similarities
        if predicted_item_features.shape[0] > 1:
            similarities = cosine_similarity(predicted_item_features)

            # Get indices for the upper right triangle without the diagonal
            upper_right = np.triu_indices(similarities.shape[0], k=1)

            # Calculate the average similarity score of all recommended itms in the list
            ils_single_user = np.mean(similarities[upper_right])

            # Calculate intra-list similarity
            avg_intralist_similarity = np.mean(ils_single_user)
        
        else:
            avg_intralist_similarity = 0
        
        # Calculate diversity
        diversity_scores[i] = 1 - (avg_intralist_similarity ** alpha)
    
    return diversity_scores

def intra_list_diversity(y_pred, imprs, embedding_matrix, topk=10):
    """    Fast intra-list diversity calculation using matrix operations."""
    topk = np.minimum(y_pred.shape[1], topk)

    # Get top-k indices 
    order = np.argsort(y_pred, axis=1)[:, ::-1][:, :topk]  
    sessions = np.take_along_axis(imprs, order, axis=1) 

    # Get embeddings
    predicted_features = embedding_matrix[sessions] 

    # Mask zero embeddings and rows with fewer than `topk` items
    valid_mask = np.linalg.norm(predicted_features, axis=-1) > 1e-6  
    valid_rows = np.sum(valid_mask, axis=1) >= topk

    # Prevent dividing by a small number
    predicted_features[~valid_mask] = np.random.normal(scale=1e-6, size=predicted_features[~valid_mask].shape)

    # Get ILD only for valid rows
    valid_indices = np.where(valid_rows)[0] 
    predicted_features_valid = predicted_features[valid_rows]  

    # Get pairwise cosine similarity
    norms = np.linalg.norm(predicted_features_valid, axis=-1, keepdims=True) 
    normalised_features = predicted_features_valid / norms
    similarities = np.matmul(normalised_features, normalised_features.transpose(0, 2, 1))  

    # Get upper triangle
    upper_right = np.triu_indices(topk, k=1)
    avg_similarity = np.mean(similarities[:, upper_right[0], upper_right[1]], axis=1)

    # Fill with Nans (use np.nanmean!)
    diversity_scores = np.full(y_pred.shape[0], np.nan, dtype=np.float32)
    diversity_scores[valid_indices] = 1 - avg_similarity
    return diversity_scores.astype('float') 

def get_clicked_imps(y_true, imprs):
    """Get all clicked news indices"""  

    pos_candidates = []
    for imp, lab in zip(imprs, y_true):   
        lab = np.array(lab)
        imp = np.array(imp)       
        mask = np.where(lab == 1)[0]
        pos_candidates.extend(list(imp[mask]))
    
    return pos_candidates

def popularity_count(imprs, labels, histories=None):
    """
    Counts the clicked items and order by number of clicks.

    As the imprs respresent the test set only, conversely, 
    all popularity measures are for the test set only.

    Returns:
        Tuple:
        A popularity counter.
        The total number of clicks.
    """  
    if histories is None: histories = []
    histories.extend(get_clicked_imps(labels, imprs))

    # Compute the most featured
    pop = Counter(histories)
    total = pop.total()
    return pop, total

def expected_pop_complement(y_true, y_pred, imprs, top_n=10):
    """
    Calculate the Expected Popularity Complement (EPC) for a set of recommendations.

    This function evaluates how well a recommendation system suggests items that are less popular among users. It does this by measuring the complement of the item's logarithmic popularity (derived from the frequency of positive interactions in the data). 
    Each recommendation in the top_n list is weighted by its position using a logarithmic discount, emphasizing higher-ranked recommendations.

    Parameters:
        y_true (iterable): Ground truth binary labels indicating the items a user has interacted with.
        y_pred (iterable): Predicted ranking scores or orders corresponding to the recommended items.
        imprs (iterable): An iterable containing arrays of item identifiers (impressions) for each recommendation instance.
        top_n (int, optional): The number of top-ranked items to consider when computing the metric (default is 5).

    Returns:
        float: The mean EPC value over all recommendation instances, rounded to four decimal places.

    References:
        Vargas, S., Castells, P., 2011. Rank and relevance in novelty and diversity metrics for recommender systems, in: Proceedings of the Fifth ACM Conference on Recommender Systems. Presented at the RecSys '11: Fifth ACM Conference on Recommender Systems, ACM, Chicago Illinois USA, pp. 109–116. https://doi.org/10.1145/2043932.2043955

    """   
    # Compute popularity counts
    pop, total = popularity_count(imprs, y_true)

    # Reduce count to a (log) probability score
    item_popularity_log = {key: np.log(count + 1) / np.log(total + 1) for key, count in pop.items()}
    
    # alt
    # item_popularity = {key: count / total for key, count in pop.items()} 

    epcs = []
    for impr, rank in zip(imprs, y_pred):
        # Rank candidates indices and take `top_n`
        order = rank.argsort()[::-1][:top_n]
        list_of_recs = np.take(impr, order)   

        # Compute EPC
        epc = 0
        for k, item in enumerate(list_of_recs):
            p_seen = item_popularity_log.get(item, 0)
            disc = 1.0 / np.log2(1 + k+1) # rank based discounting
            epc += disc * (1 - p_seen) # 1, 0.63, 0.5, 0.4
        epcs.append(epc) 
    # print("std:", round(np.std(epcs), 4))
    return round(np.mean(epcs), 4)


def surprisal(y_true: np.ndarray, 
                y_pred: np.ndarray, 
                imprs: np.ndarray, 
                top_l: int = 5) -> float:
    """
    Compute the mean top-L surprisal (novelty) metric.

    This takes a list of recommendations and uses the self-information of recommended items to compute their unexpectedness (or "surpisal") given their global popularity.

    Parameters:
        y_true (np.ndarray): 2D binary array of click labels (0 or 1) for each recommended item,
                             shape (n_users, n_recs).
        y_pred (np.ndarray): 2D array of predicted ranking scores, same shape as y_true.
        imps (np.ndarray): 2D array of recommended item indices, same shape as y_true.
        top_l (int): Number of top recommendations to consider (default: 20).

    Returns:
        float: The mean top-L surprisal (novelty) across all users.
    
    References:
        Zhou, T., Kuscsik, Z., Liu, J. G., Medo, M., Wakeling, J. R., & Zhang, Y. C. (2010).
        Solving the apparent diversity-accuracy dilemma of recommender systems.
        Proceedings of the National Academy of Sciences, 107(10), 4511-4515.
    """
    # Number of sessions 
    sessions = y_true.shape[0]

    # Total number of users (temporarily hard coded)
    u = 50000 if sessions < 1e5 else 711222

    # Get popularity count
    popularity_counter, u = popularity_count(imprs, y_true)
    
    def self_information(item):
        k = popularity_counter.get(item, 0)
        k = k if k > 0 else 1  # avoid division by zero
        return np.log2(u / k)

    # For each user, compute mean surprisal over top L items (after ranking by y_pred)
    surprisal_scores = []
    for i in range(sessions):
        # Get the scores and items for user i
        scores = y_pred[i]
        items = imprs[i]
        order = np.argsort(scores)[::-1]        
        topL = order[:min(top_l, len(order))] # if less than L, use all

        # Compute surprisal for each top-L item
        surprisal_vals = [self_information(items[j]) for j in topL]        
        if surprisal_vals:
            surprisal_scores.append(np.mean(surprisal_vals))
        else:
            surprisal_scores.append(0)
    return surprisal_scores 

def prediction_coverage(y_pred: np.ndarray, 
                        imprs: np.ndarray) -> float:
    """
    Computes the prediction coverage for a list of recommendations. 
    Assumes one click per session.

    Parameters
    ----------
    y_pred : a 2-d array of predictions
    imprs: a list 

    Returns
    ----------
    prediction_coverage:
        The prediction coverage of the recommendations as a percent
        rounded to 4 decimal places

    ----------    
    Metric Defintion:
        Ge, M., Delgado-Battenfeld, C., & Jannach, D. (2010, September).
        Beyond accuracy: evaluating recommender systems by coverage and serendipity.
        In Proceedings of the fourth ACM conference on Recommender systems (pp. 257-260). ACM.
    """    
    # Assumption: one click per sesssion
    top_n = 1

    # take all the candidates as 'catalog'
    catalog = {imp for i in imprs for imp in i if imp != 0}

    pos_candidates = []
    for imp, rank in zip(imprs, y_pred):
        click = rank.argsort()[::-1][:top_n]
        rec = np.take(imp, click)
        pos_candidates.extend(rec) 

    unique_recs = len(set(pos_candidates))
    prediction_coverage = unique_recs/len(catalog)
    return round(prediction_coverage, 4)


# -------------- Aggregation ----------------- 

def get_metrics(y_true, y_pred,
                metrics:list = ['auc', 'mean_mrr', 'ndcg@5;10']):
    """Calculate metrics.

    Available options are: 'auc', `rmse`, `logloss`, `acc` (accurary), `f1`, `mean_mrr`,
    `ndcg` (format like: ndcg@2;4;6;8), `hit` (format like: hit@2;4;6;8), `group_auc`.

    Args:
        y_true (array-like): y_true.
        preds (array-like): Predictions.
        metrics (list): List of metric names.

    Return:
        dict: Metrics.

    Examples:
        >>> cal_metric(y_true, preds, 
                        metrics=['auc', 'rmse', 'logloss', 
                        'acc', 'f1', 'mean_mrr', 'ndcg@4;8', 
                        'hit@4;8', 'group_auc'])
        {'auc': 0.4026, 'ndcg@4': 0.4953, 'ndcg@6': 0.5346, 'group_auc': 0.8096}

    References:
        Microsoft-recommenders package (see above)
    Notes:
        In `Recommenders` the NMRS model returns predicted labels and predicted rank and then uses the labels as `y_true` and rank as `y_pred`. This is wrong. We should rewrite the function to compute binary metrics to evaluate the sigmoid output and rank metrics to evaluate the softmax output. 

        The MIND leaderboard (https://msnews.github.io/) gives AUC, MRR, nDCG@5, nDCG@10.
    """
    # Convert sparse matrices to dense arrays
    if isinstance(y_true, sp.csr_matrix):
        y_true = y_true.toarray()
    if isinstance(y_pred, sp.csr_matrix):
        y_pred = y_pred.toarray()

    # Convert lists (and other formats?) to dense arrays
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true, dtype=np.float32)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred, dtype=np.float32)

    # SUPER IMPORTANT: Mask the padding
    mask = y_true != -1
    y_true = np.where(mask, y_true, 0.0) 
    y_pred = np.where(mask, y_pred, 0.0) 

    metrics_dict = {}
    for metric in metrics:

        if metric == "auc":
            auc = custom_auc(y_true, y_pred)
            metrics_dict["auc"] = round(auc, 4)       

        elif metric == "mean_mrr":
            mean_mrr = np.mean(
                [
                    mrr_score_(each_y_true, each_preds)
                    for each_y_true, each_preds in zip(y_true, y_pred)
                ]
            )
            metrics_dict["mean_mrr"] = round(mean_mrr, 4)

        elif metric.startswith("ndcg"):  # format like:  ndcg@2;4;6;8
            ndcg_list = [1, 2]
            ks = metric.split("@")
            if len(ks) > 1:
                ndcg_list = [int(token) for token in ks[1].split(";")]
            for k in ndcg_list:
                ndcg_temp = np.mean(
                    [
                        ndcg_score(each_y_true, each_preds, k)
                        for each_y_true, each_preds in zip(y_true, y_pred)
                    ]
                )
                metrics_dict[f"ndcg@{k}"] = round(ndcg_temp, 4)      
     
        else:
            raise ValueError(f"Metric {metric} not defined")  

    return metrics_dict

def get_norm_metrics(
        y_pred, 
        y_true, 
        imprs,
        norm_metrics: list=[
            "mean_epc", 
            'mean_intra_list_diversity', 
            'mean_surprisal',
            'prediction_coverage'],
        dataset_size='small', 
        **kwargs):
    """
    Calculate normative metrics.

    Available options are: `mean_intra_list_diversity`, "mean_epc", 'prediction_coverage', `mean_surprisal`.        
    Optional kwargs are: topk, top_l.

    Args:
        norm_metrics (list): List of metric names.
        y_pred (array-like): Predictions.
        **kwargs (dictionary): Dict of keyword arguments. 

    Returns:
        dict: Metrics.

    Examples:
        >>> norm_metrics = ["prediction_coverage"]
        >>> norm_kws = {'catalog': catalog}
        >>> calculate_norm_metrics(norm_metrics, preds, norm_kws)]
        {"prediction_coverage": 1.15210}
    
    """

    embedding_matrix = np.load('../.data/embedding_matrix_small_cls.npy')     
    if dataset_size == 'large':
        embedding_matrix = np.load('../.data/embedding_matrix_large_cls.npy')

    metrics_dict = {}
    for metric in norm_metrics:        
        if metric == "intra_list_diversity":            
            topk = kwargs.get('topk', 5)
            metrics_dict["intra_list_diversity"] = intra_list_diversity(y_pred, imprs, embedding_matrix, topk)
                
        elif metric == "mean_intra_list_diversity":
            topk = kwargs.get('topk', 5)
            metrics_dict["mean_intra_list_diversity"] = round(np.nanmean(intra_list_diversity(y_pred, imprs, embedding_matrix, topk)), 4)

        elif metric == "mean_epc":
            top_n = kwargs.get('top_n', 5)
            metrics_dict["mean_epc"] = np.mean(expected_pop_complement(
                y_true, y_pred, imprs, top_n))

        elif metric == "surprisal":        
            metrics_dict["surprisal"] = surprisal(y_true, y_pred, imprs)            

        elif metric == "mean_surprisal":
            top_l = kwargs.get('top_l', 5)
            metrics_dict["mean_surprisal"] = round(np.mean(surprisal(y_true, y_pred, imprs, top_l)), 4)            
        
        elif metric == "prediction_coverage":
            metrics_dict["prediction_coverage"] = prediction_coverage(y_pred, imprs)

        else:
            raise ValueError(f"Metric {metric} not defined")
    return metrics_dict

