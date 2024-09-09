import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_mean

class LightGCNAttention(MessagePassing):
    def __init__(self, num_users, num_items, embedding_dim, num_layers):
        super(LightGCNAttention, self).__init__(aggr='add')  # aggregation method: 'add'
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim

        # Embedding layers for users and items
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # Initialize embeddings
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        # Attention weight for layers
        self.attention_weight = nn.Parameter(torch.Tensor(num_layers, embedding_dim))
        nn.init.xavier_uniform_(self.attention_weight)

    def forward(self, edge_index):
        # Get user and item embeddings
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        
        # Combine user and item embeddings
        embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)  # Shape: [num_users + num_items, embedding_dim]
        
        # Initialize all_embeddings to the same size as embeddings
        all_embeddings = embeddings.clone()  # Clone the embeddings to preserve the original values

        # K-layer propagation with attention
        for layer in range(self.num_layers):
            # Propagate only the user embeddings through the graph
            user_item_embeddings = self.propagate(edge_index, x=embeddings)

            # **Important**: Manually restore the item embeddings (which may not propagate)
            # This ensures item embeddings are still included even if not propagated
            embeddings = torch.cat([user_item_embeddings[:self.num_users], item_embeddings], dim=0)

            # Debugging prints to ensure correct shapes
            print(f"Shape of embeddings after layer {layer}: {embeddings.shape}")
            print(f"Shape of attention weight at layer {layer}: {self.attention_weight[layer].shape}")
            
            # Apply attention weight with proper broadcasting
            all_embeddings += embeddings * self.attention_weight[layer].unsqueeze(0)

        # Split final embeddings back into users and items
        user_final, item_final = torch.split(all_embeddings, [self.num_users, self.num_items])
        
        return user_final, item_final

    def message(self, x_j):
        return x_j

    def aggregate(self, inputs, index):
        return scatter_mean(inputs, index, dim=0)
