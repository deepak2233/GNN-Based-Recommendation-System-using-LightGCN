import torch.nn.functional as F
import torch

def bpr_loss(user_embeddings, item_embeddings, interactions, num_users):
    user_indices = torch.tensor(interactions[:, 0], dtype=torch.long)
    pos_item_indices = torch.tensor(interactions[:, 1], dtype=torch.long)
    neg_item_indices = torch.randint(0, item_embeddings.shape[0], pos_item_indices.shape)

    user_emb = user_embeddings[user_indices]
    pos_item_emb = item_embeddings[pos_item_indices]
    neg_item_emb = item_embeddings[neg_item_indices]

    pos_scores = torch.sum(user_emb * pos_item_emb, dim=1)
    neg_scores = torch.sum(user_emb * neg_item_emb, dim=1)

    return -torch.mean(F.logsigmoid(pos_scores - neg_scores))
