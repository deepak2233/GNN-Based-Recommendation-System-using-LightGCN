"""
Tests for the LightGCN model architecture.

Covers model initialization, forward pass, embedding shapes,
and prediction functionality.
"""

import pytest
import torch
from backend.model import LightGCNAttention


@pytest.fixture
def model_params():
    """Default model parameters for testing."""
    return {
        'num_users': 10,
        'num_items': 8,
        'embedding_dim': 16,
        'num_layers': 2,
    }


@pytest.fixture
def model(model_params):
    """Create a test model instance."""
    return LightGCNAttention(**model_params)


@pytest.fixture
def edge_index(model_params):
    """Create a simple test edge index."""
    num_users = model_params['num_users']
    # Simple bipartite edges: user 0->item 0, user 1->item 1, etc.
    users = torch.arange(min(num_users, model_params['num_items']))
    items = users + num_users
    forward = torch.stack([users, items])
    backward = torch.stack([items, users])
    return torch.cat([forward, backward], dim=1)


class TestModelInit:
    """Tests for model initialization."""

    def test_model_creation(self, model, model_params):
        assert model.num_users == model_params['num_users']
        assert model.num_items == model_params['num_items']
        assert model.embedding_dim == model_params['embedding_dim']
        assert model.num_layers == model_params['num_layers']

    def test_embedding_shapes(self, model, model_params):
        assert model.user_embedding.weight.shape == (
            model_params['num_users'], model_params['embedding_dim']
        )
        assert model.item_embedding.weight.shape == (
            model_params['num_items'], model_params['embedding_dim']
        )

    def test_attention_weight_shape(self, model, model_params):
        assert model.attention_weight.shape == (
            model_params['num_layers'], model_params['embedding_dim']
        )

    def test_model_with_dropout(self, model_params):
        model = LightGCNAttention(**model_params, dropout=0.5)
        assert model.dropout == 0.5


class TestForwardPass:
    """Tests for the forward pass."""

    def test_forward_output_shapes(self, model, edge_index, model_params):
        user_emb, item_emb = model(edge_index)
        assert user_emb.shape == (model_params['num_users'], model_params['embedding_dim'])
        assert item_emb.shape == (model_params['num_items'], model_params['embedding_dim'])

    def test_forward_gradients(self, model, edge_index):
        user_emb, item_emb = model(edge_index)
        loss = user_emb.sum() + item_emb.sum()
        loss.backward()
        assert model.user_embedding.weight.grad is not None
        assert model.item_embedding.weight.grad is not None

    def test_forward_eval_mode(self, model, edge_index, model_params):
        model.eval()
        with torch.no_grad():
            user_emb, item_emb = model(edge_index)
        assert user_emb.shape == (model_params['num_users'], model_params['embedding_dim'])


class TestPrediction:
    """Tests for the prediction method."""

    def test_predict_output(self, model, edge_index):
        top_items = model.predict(0, edge_index, top_k=3)
        assert len(top_items) == 3
        assert top_items.dtype == torch.long

    def test_predict_top_k_clamped(self, model, edge_index, model_params):
        top_items = model.predict(0, edge_index, top_k=1000)
        assert len(top_items) == model_params['num_items']

    def test_predict_different_users(self, model, edge_index):
        items_0 = model.predict(0, edge_index, top_k=3)
        items_1 = model.predict(1, edge_index, top_k=3)
        # Different users should generally get different recommendations
        assert items_0.shape == items_1.shape


class TestGetEmbedding:
    """Tests for the get_embedding method."""

    def test_get_user_embeddings(self, model, edge_index):
        result = model.get_embedding(
            user_ids=torch.tensor([0, 1]), edge_index=edge_index
        )
        assert 'user_embeddings' in result
        assert result['user_embeddings'].shape[0] == 2

    def test_get_item_embeddings(self, model, edge_index):
        result = model.get_embedding(
            item_ids=torch.tensor([0, 1, 2]), edge_index=edge_index
        )
        assert 'item_embeddings' in result
        assert result['item_embeddings'].shape[0] == 3
