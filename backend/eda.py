"""
Exploratory Data Analysis for Amazon Reviews Dataset

Generates comprehensive data analysis reports and visualizations
for understanding user-item interaction patterns.
"""

import logging
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for production
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def perform_eda(file_path: str, output_dir: str = 'reports',
                plots: list = None) -> dict:
    """
    Perform comprehensive EDA on the Amazon Reviews dataset.

    Args:
        file_path: Path to the CSV data file.
        output_dir: Directory to save plots and reports.
        plots: List of plot types to generate. If None, generates all.

    Returns:
        Dictionary containing computed statistics.
    """
    file_path = Path(file_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    logger.info(f"Performing EDA on {file_path}")

    # Load dataset
    df = pd.read_csv(
        file_path, header=None,
        names=['reviewerID', 'asin', 'overall', 'timestamp']
    )

    if df.empty:
        raise ValueError("Dataset is empty")

    # Available plot generators
    available_plots = {
        'ratings_distribution': _plot_ratings_distribution,
        'user_activity': _plot_user_activity,
        'item_popularity': _plot_item_popularity,
        'interaction_timeline': _plot_interaction_timeline,
    }

    if plots is None:
        plots = list(available_plots.keys())

    # Compute statistics
    stats = _compute_statistics(df)

    # Log summary statistics
    logger.info("=== Dataset Statistics ===")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")

    # Save statistics report
    _save_stats_report(stats, output_dir / 'statistics.txt')

    # Generate plots
    for plot_name in plots:
        if plot_name in available_plots:
            try:
                save_path = output_dir / f"{plot_name}.png"
                available_plots[plot_name](df, save_path)
                logger.info(f"Plot saved: {save_path}")
            except Exception as e:
                logger.error(f"Failed to generate plot '{plot_name}': {e}")
        else:
            logger.warning(f"Unknown plot type: {plot_name}")

    return stats


def _compute_statistics(df: pd.DataFrame) -> dict:
    """Compute comprehensive dataset statistics."""
    stats = {
        'total_interactions': len(df),
        'num_users': df['reviewerID'].nunique(),
        'num_items': df['asin'].nunique(),
        'density': len(df) / (df['reviewerID'].nunique() * df['asin'].nunique()) * 100,
        'avg_rating': df['overall'].mean(),
        'median_rating': df['overall'].median(),
        'rating_std': df['overall'].std(),
        'avg_interactions_per_user': len(df) / df['reviewerID'].nunique(),
        'avg_interactions_per_item': len(df) / df['asin'].nunique(),
        'min_user_interactions': df['reviewerID'].value_counts().min(),
        'max_user_interactions': df['reviewerID'].value_counts().max(),
        'min_item_interactions': df['asin'].value_counts().min(),
        'max_item_interactions': df['asin'].value_counts().max(),
    }
    return stats


def _save_stats_report(stats: dict, path: Path) -> None:
    """Save statistics report to a text file."""
    with open(path, 'w') as f:
        f.write("LightGCN Dataset Statistics Report\n")
        f.write("=" * 50 + "\n\n")
        for key, value in stats.items():
            formatted_key = key.replace('_', ' ').title()
            if isinstance(value, float):
                f.write(f"{formatted_key}: {value:.4f}\n")
            else:
                f.write(f"{formatted_key}: {value}\n")
    logger.info(f"Statistics report saved to {path}")


def _plot_ratings_distribution(df: pd.DataFrame, save_path: Path) -> None:
    """Plot the distribution of ratings."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['overall'], kde=False, bins=5, ax=ax, color='steelblue')
    ax.set_title('Ratings Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Rating', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)

    # Add count labels on bars
    for p in ax.patches:
        ax.annotate(
            f'{int(p.get_height()):,}',
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='bottom', fontsize=10
        )

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_user_activity(df: pd.DataFrame, save_path: Path) -> None:
    """Plot distribution of user interaction counts."""
    user_counts = df['reviewerID'].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Histogram
    axes[0].hist(user_counts.values, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].set_title('User Activity Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Number of Interactions', fontsize=12)
    axes[0].set_ylabel('Number of Users', fontsize=12)
    axes[0].set_yscale('log')

    # CDF
    sorted_counts = sorted(user_counts.values)
    cdf = [i / len(sorted_counts) for i in range(1, len(sorted_counts) + 1)]
    axes[1].plot(sorted_counts, cdf, color='steelblue', linewidth=2)
    axes[1].set_title('User Activity CDF', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Number of Interactions', fontsize=12)
    axes[1].set_ylabel('Cumulative Fraction of Users', fontsize=12)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_item_popularity(df: pd.DataFrame, save_path: Path) -> None:
    """Plot distribution of item popularity (interaction counts)."""
    item_counts = df['asin'].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Histogram
    axes[0].hist(item_counts.values, bins=50, color='coral', edgecolor='black', alpha=0.7)
    axes[0].set_title('Item Popularity Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Number of Interactions', fontsize=12)
    axes[0].set_ylabel('Number of Items', fontsize=12)
    axes[0].set_yscale('log')

    # Top 20 items
    top_items = item_counts.head(20)
    axes[1].barh(range(len(top_items)), top_items.values, color='coral', alpha=0.7)
    axes[1].set_yticks(range(len(top_items)))
    axes[1].set_yticklabels([f"Item {i}" for i in range(len(top_items))], fontsize=9)
    axes[1].set_title('Top 20 Most Popular Items', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Number of Interactions', fontsize=12)
    axes[1].invert_yaxis()

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_interaction_timeline(df: pd.DataFrame, save_path: Path) -> None:
    """Plot interaction volume over time."""
    if 'timestamp' not in df.columns:
        logger.warning("No timestamp column found, skipping timeline plot")
        return

    try:
        df_time = df.copy()
        df_time['timestamp'] = pd.to_numeric(df_time['timestamp'], errors='coerce')
        df_time.dropna(subset=['timestamp'], inplace=True)

        if df_time.empty:
            logger.warning("No valid timestamps, skipping timeline plot")
            return

        df_time['date'] = pd.to_datetime(df_time['timestamp'], unit='s')
        daily_counts = df_time.groupby(df_time['date'].dt.to_period('M')).size()

        fig, ax = plt.subplots(figsize=(14, 6))
        daily_counts.plot(kind='bar', ax=ax, color='steelblue', alpha=0.7)
        ax.set_title('Interactions Over Time (Monthly)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Number of Interactions', fontsize=12)

        # Show fewer x-tick labels
        tick_count = len(daily_counts)
        if tick_count > 20:
            step = tick_count // 20
            ax.set_xticks(range(0, tick_count, step))

        plt.xticks(rotation=45)
        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        logger.error(f"Failed to plot interaction timeline: {e}")


if __name__ == '__main__':
    from backend.config_manager import get_config, setup_logging

    config = get_config()
    setup_logging(config)

    stats = perform_eda(
        file_path=str(config.get_data_path()),
        output_dir=str(config.get_eda_output_dir()),
        plots=config.eda.plots
    )
