"""
Data Loading and Preprocessing for LightGCN

Handles loading Amazon Reviews dataset, preprocessing interactions,
building bipartite graph edge indices, and train/test splitting.
"""

import logging
import hashlib
import pickle
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


def load_amazon_reviews(file_path: str, cache_dir: Optional[str] = None,
                        min_interactions: int = 0) -> Tuple[np.ndarray, int, int]:
    """
    Load and preprocess the Amazon Reviews dataset.

    Args:
        file_path: Path to the CSV file.
        cache_dir: Optional directory to cache processed data.
        min_interactions: Minimum number of interactions per user/item to keep.

    Returns:
        Tuple of (interactions_array, num_users, num_items).

    Raises:
        FileNotFoundError: If the data file doesn't exist.
        ValueError: If the data file is empty or has invalid format.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    # Check cache
    if cache_dir:
        cache_path = _get_cache_path(file_path, cache_dir, min_interactions)
        if cache_path.exists():
            logger.info(f"Loading cached data from {cache_path}")
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
            return cached['interactions'], cached['num_users'], cached['num_items']

    logger.info(f"Loading dataset from {file_path}")

    # Load the dataset
    try:
        df = pd.read_csv(
            file_path, header=None,
            names=['reviewerID', 'asin', 'overall', 'timestamp']
        )
    except Exception as e:
        raise ValueError(f"Failed to read data file: {e}")

    if df.empty:
        raise ValueError("Data file is empty")

    logger.info(f"Loaded {len(df)} raw interactions")

    # Keep relevant columns
    df = df[['reviewerID', 'asin', 'overall']].copy()

    # Remove rows with missing values
    initial_len = len(df)
    df.dropna(inplace=True)
    if len(df) < initial_len:
        logger.warning(f"Dropped {initial_len - len(df)} rows with missing values")

    # Filter by minimum interactions if specified
    if min_interactions > 0:
        df = _filter_by_min_interactions(df, min_interactions)

    # Convert user and item IDs to categorical codes
    df['userID'] = df['reviewerID'].astype('category').cat.codes
    df['itemID'] = df['asin'].astype('category').cat.codes

    # Build interactions array
    interactions = df[['userID', 'itemID', 'overall']].values
    num_users = df['userID'].nunique()
    num_items = df['itemID'].nunique()

    logger.info(
        f"Processed data: {num_users} users, {num_items} items, "
        f"{len(interactions)} interactions"
    )

    # Save to cache
    if cache_dir:
        _save_cache(cache_path, interactions, num_users, num_items)

    return interactions, num_users, num_items


def build_edge_index(interactions: np.ndarray, num_users: int) -> torch.Tensor:
    """
    Build the edge index for PyTorch Geometric from user-item interactions.

    Creates a bipartite graph where user nodes are [0, num_users) and
    item nodes are [num_users, num_users + num_items).

    Args:
        interactions: Array of shape [N, 3] with columns [userID, itemID, rating].
        num_users: Number of unique users.

    Returns:
        Edge index tensor of shape [2, 2*num_interactions] (bidirectional edges).

    Raises:
        ValueError: If interactions array has invalid shape.
    """
    if interactions.ndim != 2 or interactions.shape[1] < 2:
        raise ValueError(
            f"Expected interactions of shape [N, >=2], got {interactions.shape}"
        )

    user_indices = torch.tensor(interactions[:, 0], dtype=torch.long)
    item_indices = torch.tensor(interactions[:, 1], dtype=torch.long) + num_users

    # Create bidirectional edges for undirected graph
    forward_edges = torch.stack([user_indices, item_indices], dim=0)
    backward_edges = torch.stack([item_indices, user_indices], dim=0)
    edge_index = torch.cat([forward_edges, backward_edges], dim=1)

    logger.info(f"Built edge index with {edge_index.shape[1]} edges (bidirectional)")

    return edge_index


def train_test_split(interactions: np.ndarray, test_ratio: float = 0.1,
                     val_ratio: float = 0.1, seed: int = 42) -> dict:
    """
    Split interactions into train, validation, and test sets.

    Uses a leave-last-out strategy per user for test, and random split for validation.

    Args:
        interactions: Array of shape [N, 3] with columns [userID, itemID, rating].
        test_ratio: Fraction of interactions for testing.
        val_ratio: Fraction of interactions for validation.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with 'train', 'val', 'test' interaction arrays.
    """
    rng = np.random.RandomState(seed)

    num_interactions = len(interactions)
    indices = rng.permutation(num_interactions)

    test_size = int(num_interactions * test_ratio)
    val_size = int(num_interactions * val_ratio)

    test_indices = indices[:test_size]
    val_indices = indices[test_size:test_size + val_size]
    train_indices = indices[test_size + val_size:]

    split = {
        'train': interactions[train_indices],
        'val': interactions[val_indices],
        'test': interactions[test_indices],
    }

    logger.info(
        f"Data split: train={len(split['train'])}, val={len(split['val'])}, "
        f"test={len(split['test'])}"
    )

    return split


def _filter_by_min_interactions(df: pd.DataFrame, min_count: int) -> pd.DataFrame:
    """Filter users and items with fewer than min_count interactions."""
    prev_len = len(df)
    while True:
        user_counts = df['reviewerID'].value_counts()
        item_counts = df['asin'].value_counts()
        valid_users = user_counts[user_counts >= min_count].index
        valid_items = item_counts[item_counts >= min_count].index
        df = df[df['reviewerID'].isin(valid_users) & df['asin'].isin(valid_items)]
        if len(df) == prev_len:
            break
        prev_len = len(df)
    logger.info(f"Filtered to {len(df)} interactions (min_interactions={min_count})")
    return df


def _get_cache_path(file_path: Path, cache_dir: str, min_interactions: int) -> Path:
    """Generate a cache file path based on data file and parameters."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    file_hash = hashlib.md5(str(file_path.resolve()).encode()).hexdigest()[:8]
    cache_name = f"data_{file_hash}_min{min_interactions}.pkl"
    return cache_dir / cache_name


def _save_cache(cache_path: Path, interactions: np.ndarray,
                num_users: int, num_items: int) -> None:
    """Save processed data to cache."""
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump({
                'interactions': interactions,
                'num_users': num_users,
                'num_items': num_items,
            }, f)
        logger.info(f"Cached processed data to {cache_path}")
    except Exception as e:
        logger.warning(f"Failed to save cache: {e}")
