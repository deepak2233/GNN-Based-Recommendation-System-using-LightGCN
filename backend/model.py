"""
LightGCN with Attention - Production Model

Implements the LightGCN architecture with learned attention weights
for multi-layer graph convolution on user-item bipartite graphs.
"""

import logging
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_mean

logger = logging.getLogger(__name__)


class LightGCNAttention(MessagePassing):
    """
    LightGCN model with attention-based layer aggregation.

    This model performs multi-layer graph convolution on a user-item
    bipartite graph and uses learned attention weights to aggregate
    embeddings from different layers.

    Args:
        num_users: Number of unique users in the dataset.
        num_items: Number of unique items in the dataset.
        embedding_dim: Dimension of user/item embeddings.
        num_layers: Number of GCN propagation layers.
        dropout: Dropout rate for regularization during training.
    """

    def __init__(self, num_users: int, num_items: int, embedding_dim: int,
                 num_layers: int, dropout: float = 0.0):
        super(LightGCNAttention, self).__init__(aggr='add')
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.dropout = dropout

        # Embedding layers for users and items
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # Initialize embeddings with Xavier uniform
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        # Attention weights for layer aggregation
        self.attention_weight = nn.Parameter(torch.Tensor(num_layers, embedding_dim))
        nn.init.xavier_uniform_(self.attention_weight)

        # Dropout layer
        self.dropout_layer = nn.Dropout(p=dropout)

        logger.info(
            f"Initialized LightGCNAttention: users={num_users}, items={num_items}, "
            f"dim={embedding_dim}, layers={num_layers}, dropout={dropout}"
        )

    def forward(self, edge_index: torch.Tensor) -> tuple:
        """
        Forward pass through the LightGCN model.

        Args:
            edge_index: Graph edge index tensor of shape [2, num_edges].

        Returns:
            Tuple of (user_embeddings, item_embeddings) after multi-layer propagation.
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight

        # Combine user and item embeddings for graph propagation
        embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)

        # Initialize accumulated embeddings
        all_embeddings = embeddings.clone()

        # K-layer propagation with attention
        for layer in range(self.num_layers):
            # Propagate embeddings through the graph
            user_item_embeddings = self.propagate(edge_index, x=embeddings)

            # Restore item embeddings (items may not receive messages from all users)
            embeddings = torch.cat(
                [user_item_embeddings[:self.num_users], item_embeddings], dim=0
            )

            # Apply dropout during training
            if self.training and self.dropout > 0:
                embeddings = self.dropout_layer(embeddings)

            # Apply attention weight with proper broadcasting
            all_embeddings += embeddings * self.attention_weight[layer].unsqueeze(0)

        # Split final embeddings back into users and items
        user_final, item_final = torch.split(
            all_embeddings, [self.num_users, self.num_items]
        )

        return user_final, item_final

    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        """Message function: pass neighbor features."""
        return x_j

    def aggregate(self, inputs: torch.Tensor, index: torch.Tensor, **kwargs) -> torch.Tensor:
        """Aggregate neighbor messages using mean aggregation."""
        return scatter_mean(inputs, index, dim=0)

    def get_embedding(self, user_ids: torch.Tensor = None,
                      item_ids: torch.Tensor = None,
                      edge_index: torch.Tensor = None) -> dict:
        """
        Get embeddings for specific users and/or items.

        Args:
            user_ids: Optional tensor of user IDs to get embeddings for.
            item_ids: Optional tensor of item IDs to get embeddings for.
            edge_index: Graph edge index for forward pass.

        Returns:
            Dictionary with 'user_embeddings' and/or 'item_embeddings'.
        """
        result = {}
        with torch.no_grad():
            user_emb, item_emb = self.forward(edge_index)
            if user_ids is not None:
                result['user_embeddings'] = user_emb[user_ids]
            if item_ids is not None:
                result['item_embeddings'] = item_emb[item_ids]
        return result

    def predict(self, user_id: int, edge_index: torch.Tensor, top_k: int = 10) -> torch.Tensor:
        """
        Generate top-K item recommendations for a single user.

        Args:
            user_id: The user ID to generate recommendations for.
            edge_index: Graph edge index tensor.
            top_k: Number of recommendations to return.

        Returns:
            Tensor of top-K recommended item indices.
        """
        self.eval()
        with torch.no_grad():
            user_emb, item_emb = self.forward(edge_index)
            scores = torch.matmul(user_emb[user_id], item_emb.T)
            _, top_items = torch.topk(scores, min(top_k, self.num_items))
        return top_items
