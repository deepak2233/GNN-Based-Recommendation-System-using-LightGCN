"""
Tests for data loading and preprocessing functions.
"""

import pytest
import numpy as np
import torch
from backend.data_loader import build_edge_index, train_test_split


class TestBuildEdgeIndex:
    """Tests for the build_edge_index function."""

    def test_edge_index_shape(self):
        interactions = np.array([[0, 0, 5.0], [1, 1, 4.0], [2, 2, 3.0]])
        edge_index = build_edge_index(interactions, num_users=3)
        # Bidirectional: 2 * 3 = 6 edges
        assert edge_index.shape == (2, 6)

    def test_edge_index_dtype(self):
        interactions = np.array([[0, 0, 5.0]])
        edge_index = build_edge_index(interactions, num_users=1)
        assert edge_index.dtype == torch.long

    def test_edge_index_item_offset(self):
        interactions = np.array([[0, 0, 5.0]])
        num_users = 5
        edge_index = build_edge_index(interactions, num_users=num_users)
        # Item index should be shifted by num_users
        assert edge_index[1, 0].item() == num_users  # item 0 + 5 = 5

    def test_edge_index_bidirectional(self):
        interactions = np.array([[0, 0, 5.0]])
        num_users = 3
        edge_index = build_edge_index(interactions, num_users=num_users)
        # Forward: [0, 3], Backward: [3, 0]
        assert edge_index.shape[1] == 2
        assert edge_index[0, 0].item() == 0
        assert edge_index[1, 0].item() == 3
        assert edge_index[0, 1].item() == 3
        assert edge_index[1, 1].item() == 0

    def test_edge_index_invalid_shape(self):
        interactions = np.array([0, 1, 2])  # 1D array
        with pytest.raises(ValueError):
            build_edge_index(interactions, num_users=1)


class TestTrainTestSplit:
    """Tests for the train_test_split function."""

    def test_split_sizes(self):
        interactions = np.random.rand(100, 3)
        split = train_test_split(interactions, test_ratio=0.2, val_ratio=0.1)
        assert len(split['test']) == 20
        assert len(split['val']) == 10
        assert len(split['train']) == 70

    def test_split_no_overlap(self):
        interactions = np.arange(300).reshape(100, 3)
        split = train_test_split(interactions, test_ratio=0.2, val_ratio=0.1)
        train_set = set(map(tuple, split['train']))
        val_set = set(map(tuple, split['val']))
        test_set = set(map(tuple, split['test']))
        assert len(train_set & val_set) == 0
        assert len(train_set & test_set) == 0
        assert len(val_set & test_set) == 0

    def test_split_reproducibility(self):
        interactions = np.random.rand(100, 3)
        split1 = train_test_split(interactions, seed=42)
        split2 = train_test_split(interactions, seed=42)
        np.testing.assert_array_equal(split1['train'], split2['train'])

    def test_split_total_preserves_data(self):
        interactions = np.random.rand(100, 3)
        split = train_test_split(interactions, test_ratio=0.2, val_ratio=0.1)
        total = len(split['train']) + len(split['val']) + len(split['test'])
        assert total == 100
