"""
Evaluation Metrics for LightGCN Recommendation System

Implements standard recommendation system evaluation metrics:
Hit Ratio, NDCG, Precision, Recall, and MRR at various K values.
"""

import logging
from typing import Dict, List

import numpy as np
import torch

logger = logging.getLogger(__name__)


def hit_ratio_at_k(ranking: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    """
    Hit Ratio at K (HR@K).

    Checks if the ground truth item is within the top K recommendations.

    Args:
        ranking: Array of shape [num_users, K] with recommended item indices.
        ground_truth: Array of ground truth item indices per user.
        k: Number of top recommendations to consider.

    Returns:
        Hit ratio score between 0 and 1.
    """
    hits = 0
    for idx, gt_item in enumerate(ground_truth):
        if gt_item in ranking[idx, :k]:
            hits += 1
    return hits / len(ground_truth)


def ndcg_at_k(ranking: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    """
    Normalized Discounted Cumulative Gain at K (NDCG@K).

    Measures ranking quality based on the position of the relevant item.

    Args:
        ranking: Array of shape [num_users, K] with recommended item indices.
        ground_truth: Array of ground truth item indices per user.
        k: Number of top recommendations to consider.

    Returns:
        NDCG score between 0 and 1.
    """
    ndcg = 0.0
    for idx, gt_item in enumerate(ground_truth):
        if gt_item in ranking[idx, :k]:
            rank_position = np.where(ranking[idx, :k] == gt_item)[0][0]
            ndcg += 1 / np.log2(rank_position + 2)
    return ndcg / len(ground_truth)


def precision_at_k(ranking: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    """
    Precision at K.

    Fraction of recommended items in top-K that are relevant.

    Args:
        ranking: Array of shape [num_users, K] with recommended item indices.
        ground_truth: Array of ground truth item indices per user.
        k: Number of top recommendations to consider.

    Returns:
        Precision score between 0 and 1.
    """
    precision = 0.0
    for idx, gt_item in enumerate(ground_truth):
        if gt_item in ranking[idx, :k]:
            precision += 1.0 / k
    return precision / len(ground_truth)


def recall_at_k(ranking: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    """
    Recall at K.

    Fraction of relevant items that appear in top-K recommendations.

    Args:
        ranking: Array of shape [num_users, K] with recommended item indices.
        ground_truth: Array of ground truth item indices per user.
        k: Number of top recommendations to consider.

    Returns:
        Recall score between 0 and 1.
    """
    recall = 0.0
    for idx, gt_item in enumerate(ground_truth):
        if gt_item in ranking[idx, :k]:
            recall += 1.0
    return recall / len(ground_truth)


def mrr_at_k(ranking: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    """
    Mean Reciprocal Rank at K (MRR@K).

    Average of the reciprocal rank of the first relevant item.

    Args:
        ranking: Array of shape [num_users, K] with recommended item indices.
        ground_truth: Array of ground truth item indices per user.
        k: Number of top recommendations to consider.

    Returns:
        MRR score between 0 and 1.
    """
    mrr = 0.0
    for idx, gt_item in enumerate(ground_truth):
        if gt_item in ranking[idx, :k]:
            rank_position = np.where(ranking[idx, :k] == gt_item)[0][0]
            mrr += 1.0 / (rank_position + 1)
    return mrr / len(ground_truth)


METRIC_FUNCTIONS = {
    'hit_ratio': hit_ratio_at_k,
    'ndcg': ndcg_at_k,
    'precision': precision_at_k,
    'recall': recall_at_k,
    'mrr': mrr_at_k,
}


def evaluate(model, edge_index: torch.Tensor, interactions: np.ndarray,
             num_users: int, num_items: int, top_k: int = 10,
             metrics: List[str] = None) -> Dict[str, float]:
    """
    Evaluate the model using multiple recommendation metrics.

    Args:
        model: Trained LightGCN model.
        edge_index: Graph edge index tensor.
        interactions: Interaction array with columns [userID, itemID, rating].
        num_users: Number of unique users.
        num_items: Number of unique items.
        top_k: Number of top recommendations to evaluate.
        metrics: List of metric names to compute. If None, uses all available.

    Returns:
        Dictionary mapping metric names (e.g., 'hr@10') to their scores.
    """
    if metrics is None:
        metrics = list(METRIC_FUNCTIONS.keys())

    model.eval()
    results = {}

    with torch.no_grad():
        user_embeddings, item_embeddings = model(edge_index)

        # Calculate scores for all user-item pairs
        scores = torch.matmul(user_embeddings, item_embeddings.T)

        # Get top K recommended items for each user
        _, ranking = torch.topk(scores, min(top_k, num_items))
        ranking_np = ranking.cpu().numpy()

        # Extract ground truth from interactions
        ground_truth = interactions[:, 1].astype(int)

        # Compute each metric
        for metric_name in metrics:
            if metric_name in METRIC_FUNCTIONS:
                metric_fn = METRIC_FUNCTIONS[metric_name]
                score = metric_fn(ranking_np, ground_truth, top_k)
                key = f"{metric_name}@{top_k}"
                results[key] = score

    return results


def evaluate_multi_k(model, edge_index: torch.Tensor, interactions: np.ndarray,
                     num_users: int, num_items: int,
                     k_values: List[int] = None,
                     metrics: List[str] = None) -> Dict[str, float]:
    """
    Evaluate the model at multiple K values.

    Args:
        model: Trained LightGCN model.
        edge_index: Graph edge index tensor.
        interactions: Interaction array.
        num_users: Number of unique users.
        num_items: Number of unique items.
        k_values: List of K values to evaluate at.
        metrics: List of metric names to compute.

    Returns:
        Dictionary mapping metric names (e.g., 'hr@5', 'ndcg@10') to scores.
    """
    if k_values is None:
        k_values = [5, 10, 20]
    if metrics is None:
        metrics = list(METRIC_FUNCTIONS.keys())

    all_results = {}
    max_k = max(k_values)

    model.eval()
    with torch.no_grad():
        user_embeddings, item_embeddings = model(edge_index)
        scores = torch.matmul(user_embeddings, item_embeddings.T)
        _, ranking = torch.topk(scores, min(max_k, num_items))
        ranking_np = ranking.cpu().numpy()
        ground_truth = interactions[:, 1].astype(int)

        for k in k_values:
            for metric_name in metrics:
                if metric_name in METRIC_FUNCTIONS:
                    metric_fn = METRIC_FUNCTIONS[metric_name]
                    score = metric_fn(ranking_np, ground_truth, k)
                    key = f"{metric_name}@{k}"
                    all_results[key] = score

    # Log results
    for key, value in sorted(all_results.items()):
        logger.info(f"  {key}: {value:.4f}")

    return all_results
