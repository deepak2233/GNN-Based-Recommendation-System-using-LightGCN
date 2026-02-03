"""
LightGCN Recommendation System - Main Entry Point

Production-ready CLI for running EDA, training, evaluation, and serving.

Usage:
    python main.py --mode eda        # Run exploratory data analysis
    python main.py --mode train      # Train the LightGCN model
    python main.py --mode evaluate   # Evaluate the trained model
    python main.py --mode serve      # Start the FastAPI server
    python main.py --mode pipeline   # Run full pipeline (EDA -> train -> evaluate)
"""

import argparse
import logging
import sys

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="LightGCN Recommendation System Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode train
  python main.py --mode serve
  python main.py --mode evaluate --top-k 20
  python main.py --mode pipeline
  python main.py --mode train --config custom_config.yaml
        """
    )

    parser.add_argument(
        '--mode', type=str, required=True,
        choices=['eda', 'train', 'evaluate', 'serve', 'pipeline'],
        help='Pipeline mode to run'
    )
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to custom config file (default: backend/config.yaml)'
    )
    parser.add_argument(
        '--top-k', type=int, default=None,
        help='Top-K value for evaluation (overrides config)'
    )
    parser.add_argument(
        '--epochs', type=int, default=None,
        help='Number of training epochs (overrides config)'
    )
    parser.add_argument(
        '--host', type=str, default=None,
        help='API server host (overrides config)'
    )
    parser.add_argument(
        '--port', type=int, default=None,
        help='API server port (overrides config)'
    )

    return parser.parse_args()


def run_eda(config):
    """Run exploratory data analysis."""
    from backend.eda import perform_eda

    logger.info("=" * 60)
    logger.info("Starting Exploratory Data Analysis")
    logger.info("=" * 60)

    stats = perform_eda(
        file_path=str(config.get_data_path()),
        output_dir=str(config.get_eda_output_dir()),
        plots=config.eda.plots
    )

    logger.info("EDA complete. Reports saved to: %s", config.get_eda_output_dir())
    return stats


def run_training(config):
    """Run model training pipeline."""
    import torch.optim as optim
    from backend.model import LightGCNAttention
    from backend.data_loader import load_amazon_reviews, build_edge_index, train_test_split
    from backend.train import train_lightgcn

    logger.info("=" * 60)
    logger.info("Starting Model Training")
    logger.info("=" * 60)

    # Load data
    data_path = str(config.get_data_path())
    cache_dir = str(config.get_cache_dir())

    logger.info("Loading data from: %s", data_path)
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

    # Build graph
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

    logger.info("Training complete. Model saved to: %s", model_path)
    return history


def run_evaluation(config, top_k_override=None):
    """Run model evaluation."""
    from backend.data_loader import load_amazon_reviews, build_edge_index, train_test_split
    from backend.evaluate import evaluate_multi_k
    from backend.utils import load_model

    logger.info("=" * 60)
    logger.info("Starting Model Evaluation")
    logger.info("=" * 60)

    # Load data
    data_path = str(config.get_data_path())
    cache_dir = str(config.get_cache_dir())

    interactions, num_users, num_items = load_amazon_reviews(
        data_path, cache_dir=cache_dir,
        min_interactions=config.data.min_interactions
    )

    # Split data (use same split as training)
    split = train_test_split(
        interactions,
        test_ratio=config.training.test_split,
        val_ratio=config.training.validation_split
    )

    edge_index = build_edge_index(split['train'], num_users)

    # Load model
    model_path = str(config.get_model_save_path())
    model, metadata = load_model(model_path)
    logger.info("Loaded model with metadata: %s", metadata)

    # Evaluate
    k_values = config.evaluation.top_k_values
    if top_k_override:
        k_values = [top_k_override]

    results = evaluate_multi_k(
        model, edge_index, split['test'],
        num_users, num_items,
        k_values=k_values,
        metrics=config.evaluation.metrics
    )

    logger.info("Evaluation Results:")
    for metric, score in sorted(results.items()):
        logger.info("  %s: %.4f", metric, score)

    return results


def run_serve(config, host_override=None, port_override=None):
    """Start the API server."""
    import uvicorn

    host = host_override or config.api.host
    port = port_override or config.api.port

    logger.info("=" * 60)
    logger.info("Starting API Server on %s:%d", host, port)
    logger.info("=" * 60)

    uvicorn.run(
        "backend.app:app",
        host=host,
        port=port,
        workers=config.api.workers,
        reload=config.api.reload,
        log_level=config.logging.level.lower(),
    )


def run_pipeline(config):
    """Run the full pipeline: EDA -> Train -> Evaluate."""
    logger.info("=" * 60)
    logger.info("Running Full Pipeline")
    logger.info("=" * 60)

    # Step 1: EDA
    logger.info("[1/3] Running EDA...")
    run_eda(config)

    # Step 2: Training
    logger.info("[2/3] Running Training...")
    run_training(config)

    # Step 3: Evaluation
    logger.info("[3/3] Running Evaluation...")
    results = run_evaluation(config)

    logger.info("=" * 60)
    logger.info("Pipeline Complete!")
    logger.info("=" * 60)
    return results


def main():
    """Main entry point."""
    args = parse_args()

    # Load configuration
    from backend.config_manager import ConfigManager, setup_logging

    manager = ConfigManager()
    config = manager.load_config(args.config)

    # Apply CLI overrides
    if args.epochs:
        config.training.epochs = args.epochs

    # Setup logging
    setup_logging(config)

    logger.info("LightGCN Recommendation System v1.0.0")
    logger.info("Mode: %s", args.mode)

    try:
        if args.mode == 'eda':
            run_eda(config)
        elif args.mode == 'train':
            run_training(config)
        elif args.mode == 'evaluate':
            run_evaluation(config, top_k_override=args.top_k)
        elif args.mode == 'serve':
            run_serve(config, host_override=args.host, port_override=args.port)
        elif args.mode == 'pipeline':
            run_pipeline(config)
    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error("Pipeline failed: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
