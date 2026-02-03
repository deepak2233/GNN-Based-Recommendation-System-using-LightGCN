"""
Tests for evaluation metrics.
"""

import pytest
import numpy as np
from backend.evaluate import (
    hit_ratio_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    mrr_at_k,
)


@pytest.fixture
def perfect_ranking():
    """Ranking where ground truth is always at position 0."""
    ranking = np.array([[5, 3, 7], [2, 8, 1], [9, 4, 6]])
    ground_truth = np.array([5, 2, 9])
    return ranking, ground_truth


@pytest.fixture
def partial_ranking():
    """Ranking where ground truth is only sometimes in top-K."""
    ranking = np.array([[5, 3, 7], [2, 8, 1], [9, 4, 6]])
    ground_truth = np.array([5, 0, 6])  # 5 at pos 0, 0 not in top-3, 6 at pos 2
    return ranking, ground_truth


class TestHitRatio:
    def test_perfect_hit_ratio(self, perfect_ranking):
        ranking, gt = perfect_ranking
        assert hit_ratio_at_k(ranking, gt, 3) == 1.0

    def test_partial_hit_ratio(self, partial_ranking):
        ranking, gt = partial_ranking
        assert hit_ratio_at_k(ranking, gt, 3) == pytest.approx(2 / 3)

    def test_zero_hit_ratio(self):
        ranking = np.array([[1, 2, 3]])
        gt = np.array([99])
        assert hit_ratio_at_k(ranking, gt, 3) == 0.0


class TestNDCG:
    def test_perfect_ndcg(self, perfect_ranking):
        ranking, gt = perfect_ranking
        # All items at position 0: NDCG = 1/log2(2) = 1.0 per item
        assert ndcg_at_k(ranking, gt, 3) == pytest.approx(1.0)

    def test_zero_ndcg(self):
        ranking = np.array([[1, 2, 3]])
        gt = np.array([99])
        assert ndcg_at_k(ranking, gt, 3) == 0.0


class TestPrecision:
    def test_perfect_precision(self, perfect_ranking):
        ranking, gt = perfect_ranking
        # 1 relevant / 3 recommended per user = 1/3 each
        assert precision_at_k(ranking, gt, 3) == pytest.approx(1 / 3)


class TestRecall:
    def test_perfect_recall(self, perfect_ranking):
        ranking, gt = perfect_ranking
        assert recall_at_k(ranking, gt, 3) == 1.0


class TestMRR:
    def test_perfect_mrr(self, perfect_ranking):
        ranking, gt = perfect_ranking
        # All at rank 1, so MRR = 1.0
        assert mrr_at_k(ranking, gt, 3) == pytest.approx(1.0)

    def test_zero_mrr(self):
        ranking = np.array([[1, 2, 3]])
        gt = np.array([99])
        assert mrr_at_k(ranking, gt, 3) == 0.0
