"""
Training Pipeline for LightGCN

Production-ready training with early stopping, checkpointing,
validation monitoring, and device management.
"""

import logging
import time
from pathlib import Path
from typing import Optional

import torch
import torch.optim as optim

from backend.model import LightGCNAttention
from backend.data_loader import load_amazon_reviews, build_edge_index, train_test_split
from backend.evaluate import evaluate
from backend.utils import bpr_loss, get_device, move_to_device, save_model

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping to halt training when validation loss stops improving."""

    def __init__(self, patience: int = 3, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            logger.info(
                f"EarlyStopping counter: {self.counter}/{self.patience}"
            )
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.should_stop


def train_lightgcn(model, optimizer, edge_index, train_interactions, num_users,
                   num_items, epochs, model_path, val_interactions=None,
                   device=None, checkpoint_dir=None, patience=3,
                   reg_weight=0.0001):
    """
    Train the LightGCN model with validation monitoring and checkpointing.

    Args:
        model: LightGCNAttention model instance.
        optimizer: PyTorch optimizer.
        edge_index: Graph edge index tensor.
        train_interactions: Training interaction data.
        num_users: Number of unique users.
        num_items: Number of unique items.
        epochs: Maximum number of training epochs.
        model_path: Path to save the final model.
        val_interactions: Optional validation interaction data.
        device: Compute device (auto-detected if None).
        checkpoint_dir: Directory for epoch checkpoints.
        patience: Early stopping patience.
        reg_weight: L2 regularization weight.

    Returns:
        Dictionary with training history.
    """
    # Setup device
    if device is None:
        device = get_device()
    model, edge_index, device = move_to_device(model, edge_index, device)

    # Setup early stopping
    early_stopping = EarlyStopping(patience=patience) if val_interactions is not None else None

    # Setup checkpoint directory
    if checkpoint_dir:
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

    history = {
        'train_loss': [],
        'val_hr': [],
        'val_ndcg': [],
        'epoch_time': [],
    }

    best_val_ndcg = 0.0

    logger.info(f"Starting training: {epochs} epochs on {device}")

    for epoch in range(epochs):
        epoch_start = time.time()

        # Training step
        model.train()
        optimizer.zero_grad()

        user_embeddings, item_embeddings = model(edge_index)
        loss = bpr_loss(
            user_embeddings, item_embeddings,
            train_interactions, num_users, reg_weight=reg_weight
        )

        loss.backward()
        optimizer.step()

        epoch_time = time.time() - epoch_start
        train_loss = loss.item()
        history['train_loss'].append(train_loss)
        history['epoch_time'].append(epoch_time)

        log_msg = (
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss: {train_loss:.6f} | "
            f"Time: {epoch_time:.2f}s"
        )

        # Validation step
        if val_interactions is not None:
            val_metrics = evaluate(
                model, edge_index, val_interactions,
                num_users, num_items, top_k=10
            )
            history['val_hr'].append(val_metrics['hr@10'])
            history['val_ndcg'].append(val_metrics['ndcg@10'])
            log_msg += (
                f" | Val HR@10: {val_metrics['hr@10']:.4f}"
                f" | Val NDCG@10: {val_metrics['ndcg@10']:.4f}"
            )

            # Save best model
            if val_metrics['ndcg@10'] > best_val_ndcg:
                best_val_ndcg = val_metrics['ndcg@10']
                save_model(model, model_path, metadata={
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'val_ndcg@10': val_metrics['ndcg@10'],
                    'val_hr@10': val_metrics['hr@10'],
                })
                logger.info(f"Best model saved (NDCG@10: {best_val_ndcg:.4f})")

            # Early stopping check
            if early_stopping and early_stopping(train_loss):
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

        logger.info(log_msg)

        # Save checkpoint
        if checkpoint_dir and (epoch + 1) % 5 == 0:
            ckpt_path = str(Path(checkpoint_dir) / f"checkpoint_epoch_{epoch + 1}.pth")
            save_model(model, ckpt_path, metadata={
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'optimizer_state': optimizer.state_dict(),
            })

    # Save final model if no validation was used
    if val_interactions is None:
        save_model(model, model_path, metadata={
            'epoch': epochs,
            'train_loss': history['train_loss'][-1],
        })

    logger.info(
        f"Training complete. Best loss: {min(history['train_loss']):.6f}"
    )
    return history


if __name__ == '__main__':
    import yaml
    from backend.config_manager import get_config, setup_logging

    config = get_config()
    setup_logging(config)

    # Load data
    data_path = str(config.get_data_path())
    cache_dir = str(config.get_cache_dir())
    interactions, num_users, num_items = load_amazon_reviews(
        data_path, cache_dir=cache_dir,
        min_interactions=config.data.min_interactions
    )

    # Split data
    split = train_test_split(
        interactions,
        test_ratio=config.training.test_split,
        val_ratio=config.training.validation_split
    )

    edge_index = build_edge_index(split['train'], num_users)

    # Initialize model
    model = LightGCNAttention(
        num_users, num_items,
        config.model.embedding_dim,
        config.model.num_layers,
        dropout=config.model.dropout
    )
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )

    # Train
    model_path = str(config.get_model_save_path())
    checkpoint_dir = str(config.get_checkpoint_dir())

    history = train_lightgcn(
        model, optimizer, edge_index,
        split['train'], num_users, num_items,
        config.training.epochs, model_path,
        val_interactions=split['val'],
        checkpoint_dir=checkpoint_dir,
        patience=config.training.early_stopping_patience,
        reg_weight=config.training.weight_decay
    )
