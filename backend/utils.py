"""
Loss Functions and Utility Functions for LightGCN

Includes BPR loss with L2 regularization and helper utilities
for recommendation system training.
"""

import logging
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def bpr_loss(user_embeddings: torch.Tensor, item_embeddings: torch.Tensor,
             interactions: torch.Tensor, num_users: int,
             reg_weight: float = 0.0001) -> torch.Tensor:
    """
    Bayesian Personalized Ranking (BPR) loss with L2 regularization.

    Computes the BPR loss that encourages positive item scores to be higher
    than randomly sampled negative item scores.

    Args:
        user_embeddings: User embedding tensor of shape [num_users, dim].
        item_embeddings: Item embedding tensor of shape [num_items, dim].
        interactions: Interaction array with columns [userID, itemID, rating].
        num_users: Total number of users.
        reg_weight: L2 regularization weight.

    Returns:
        Scalar loss tensor.
    """
    if not isinstance(interactions, torch.Tensor):
        interactions = torch.tensor(interactions, dtype=torch.long)

    user_indices = interactions[:, 0].long()
    pos_item_indices = interactions[:, 1].long()

    # Sample negative items (items user hasn't interacted with)
    neg_item_indices = torch.randint(
        0, item_embeddings.shape[0], pos_item_indices.shape,
        device=item_embeddings.device
    )

    # Get embeddings
    user_emb = user_embeddings[user_indices]
    pos_item_emb = item_embeddings[pos_item_indices]
    neg_item_emb = item_embeddings[neg_item_indices]

    # Compute scores
    pos_scores = torch.sum(user_emb * pos_item_emb, dim=1)
    neg_scores = torch.sum(user_emb * neg_item_emb, dim=1)

    # BPR loss
    bpr = -torch.mean(F.logsigmoid(pos_scores - neg_scores))

    # L2 regularization
    reg_loss = reg_weight * (
        user_emb.norm(2).pow(2) +
        pos_item_emb.norm(2).pow(2) +
        neg_item_emb.norm(2).pow(2)
    ) / user_emb.shape[0]

    return bpr + reg_loss


def get_device() -> torch.device:
    """Get the best available compute device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    return device


def move_to_device(model: torch.nn.Module, edge_index: torch.Tensor,
                   device: Optional[torch.device] = None) -> tuple:
    """Move model and data to the specified device."""
    if device is None:
        device = get_device()
    model = model.to(device)
    edge_index = edge_index.to(device)
    return model, edge_index, device


def save_model(model: torch.nn.Module, path: str, metadata: Optional[dict] = None) -> None:
    """
    Save model state with optional metadata.

    Args:
        model: The model to save.
        path: Path to save the model.
        metadata: Optional metadata dict to save alongside the model.
    """
    from pathlib import Path as PathLib
    save_path = PathLib(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'num_users': model.num_users,
        'num_items': model.num_items,
        'embedding_dim': model.embedding_dim,
        'num_layers': model.num_layers,
    }
    if metadata:
        checkpoint['metadata'] = metadata

    torch.save(checkpoint, save_path)
    logger.info(f"Model saved to {save_path}")


def load_model(path: str, device: Optional[torch.device] = None) -> tuple:
    """
    Load a model from checkpoint.

    Args:
        path: Path to the saved checkpoint.
        device: Device to load the model onto.

    Returns:
        Tuple of (model, metadata).
    """
    from backend.model import LightGCNAttention

    if device is None:
        device = get_device()

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model = LightGCNAttention(
        num_users=checkpoint['num_users'],
        num_items=checkpoint['num_items'],
        embedding_dim=checkpoint['embedding_dim'],
        num_layers=checkpoint['num_layers'],
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    metadata = checkpoint.get('metadata', {})
    logger.info(f"Model loaded from {path}")

    return model, metadata
