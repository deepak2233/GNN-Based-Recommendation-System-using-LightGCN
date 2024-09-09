import torch
import numpy as np

def hit_ratio_at_k(ranking, ground_truth, k):
    """
    Hit Ratio at K (HR@K)
    Checks if the ground truth item is within the top K recommendations.
    """
    hits = 0
    for idx, gt_item in enumerate(ground_truth):
        if gt_item in ranking[idx, :k]:
            hits += 1
    return hits / len(ground_truth)

def ndcg_at_k(ranking, ground_truth, k):
    """
    NDCG at K (NDCG@K)
    Measures the quality of ranking based on the position of the relevant item in the top K list.
    """
    ndcg = 0.0
    for idx, gt_item in enumerate(ground_truth):
        if gt_item in ranking[idx, :k]:
            rank_position = np.where(ranking[idx, :k] == gt_item)[0][0]
            ndcg += 1 / np.log2(rank_position + 2)  # position is 0-based index, so +2
    return ndcg / len(ground_truth)

def evaluate(model, edge_index, interactions, num_users, num_items, top_k=10):
    """
    Evaluate the model using Hit Ratio@K and NDCG@K.
    """
    model.eval()
    with torch.no_grad():
        user_embeddings, item_embeddings = model(edge_index)

        # Calculate the scores for all user-item pairs
        scores = torch.matmul(user_embeddings, item_embeddings.T)
        
        # Get the top K recommended items for each user
        _, ranking = torch.topk(scores, top_k)

        # Ground truth (positive items that the users interacted with)
        user_indices = interactions[:, 0]
        ground_truth = interactions[:, 1]

        # Calculate evaluation metrics
        hr = hit_ratio_at_k(ranking.cpu().numpy(), ground_truth, top_k)
        ndcg = ndcg_at_k(ranking.cpu().numpy(), ground_truth, top_k)

        print(f"Hit Ratio@{top_k}: {hr:.4f}, NDCG@{top_k}: {ndcg:.4f}")
        return hr, ndcg
