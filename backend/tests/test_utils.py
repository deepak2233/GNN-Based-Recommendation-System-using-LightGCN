"""
Tests for utility functions and BPR loss.
"""

import pytest
import torch
import numpy as np
from backend.utils import bpr_loss


class TestBPRLoss:
    """Tests for the BPR loss function."""

    def test_loss_is_scalar(self):
        user_emb = torch.randn(10, 16)
        item_emb = torch.randn(8, 16)
        interactions = np.array([[0, 0, 5], [1, 1, 4], [2, 2, 3]])
        loss = bpr_loss(user_emb, item_emb, interactions, num_users=10)
        assert loss.dim() == 0  # scalar

    def test_loss_is_positive(self):
        user_emb = torch.randn(10, 16)
        item_emb = torch.randn(8, 16)
        interactions = np.array([[0, 0, 5], [1, 1, 4]])
        loss = bpr_loss(user_emb, item_emb, interactions, num_users=10)
        assert loss.item() > 0

    def test_loss_requires_grad(self):
        user_emb = torch.randn(10, 16, requires_grad=True)
        item_emb = torch.randn(8, 16, requires_grad=True)
        interactions = np.array([[0, 0, 5], [1, 1, 4]])
        loss = bpr_loss(user_emb, item_emb, interactions, num_users=10)
        loss.backward()
        assert user_emb.grad is not None

    def test_loss_with_tensor_interactions(self):
        user_emb = torch.randn(10, 16)
        item_emb = torch.randn(8, 16)
        interactions = torch.tensor([[0, 0, 5], [1, 1, 4]])
        loss = bpr_loss(user_emb, item_emb, interactions, num_users=10)
        assert loss.dim() == 0

    def test_loss_with_regularization(self):
        user_emb = torch.randn(10, 16)
        item_emb = torch.randn(8, 16)
        interactions = np.array([[0, 0, 5]])
        loss_no_reg = bpr_loss(user_emb, item_emb, interactions, 10, reg_weight=0.0)
        loss_with_reg = bpr_loss(user_emb, item_emb, interactions, 10, reg_weight=0.1)
        # Loss with regularization should be higher
        assert loss_with_reg.item() >= loss_no_reg.item()
