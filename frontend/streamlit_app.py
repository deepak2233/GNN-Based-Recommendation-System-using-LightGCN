"""
Interactive Streamlit Dashboard for LightGCN Recommendation System

Provides a rich UI for:
- Getting single and batch user recommendations
- Exploring dataset statistics and EDA visualizations
- Viewing model architecture and health status
- Comparing recommendations across users
"""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is on path for imports
_project_root = Path(__file__).parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from backend.config_manager import ConfigManager, setup_logging, get_project_root
from backend.model import LightGCNAttention
from backend.data_loader import load_amazon_reviews, build_edge_index
from backend.utils import load_model, get_device

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="LightGCN Recommender",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Helpers – cached resource loaders
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading configuration...")
def load_config():
    """Load and cache the project configuration."""
    ConfigManager._instance = None
    ConfigManager._config = None
    manager = ConfigManager()
    config = manager.load_config()
    setup_logging(config)
    return config


@st.cache_resource(show_spinner="Loading dataset...")
def load_data(_config):
    """Load and cache the dataset."""
    data_path = str(_config.get_data_path())
    cache_dir = str(_config.get_cache_dir())
    interactions, num_users, num_items = load_amazon_reviews(
        data_path, cache_dir=cache_dir
    )
    edge_index = build_edge_index(interactions, num_users)
    return interactions, num_users, num_items, edge_index


@st.cache_resource(show_spinner="Loading raw dataframe...")
def load_raw_dataframe(_config):
    """Load the raw CSV for EDA display."""
    data_path = str(_config.get_data_path())
    df = pd.read_csv(
        data_path, header=None,
        names=['reviewerID', 'asin', 'overall', 'timestamp'],
    )
    return df


@st.cache_resource(show_spinner="Loading model...")
def load_trained_model(_config, _num_users, _num_items, _device):
    """Load and cache the trained model."""
    model_path = str(_config.get_model_save_path())
    if not Path(model_path).exists():
        return None, "Model file not found. Train the model first."

    try:
        model, metadata = load_model(model_path, _device)
        return model, metadata
    except Exception:
        # Legacy format fallback
        model = LightGCNAttention(
            _num_users, _num_items,
            _config.model.embedding_dim,
            _config.model.num_layers,
        )
        state = torch.load(model_path, map_location=_device, weights_only=False)
        model.load_state_dict(state)
        model.to(_device)
        model.eval()
        return model, {}


def get_recommendations(model, edge_index, user_id, top_k, device):
    """Generate recommendations for a single user."""
    edge_index_dev = edge_index.to(device)
    model.eval()
    with torch.no_grad():
        user_emb, item_emb = model(edge_index_dev)
        scores = torch.matmul(user_emb[user_id], item_emb.T)
        top_scores, top_items = torch.topk(scores, top_k)
    return top_items.cpu().tolist(), top_scores.cpu().tolist()


def get_user_history(interactions, user_id):
    """Return the items a user has interacted with."""
    mask = interactions[:, 0] == user_id
    user_items = interactions[mask]
    return user_items


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("LightGCN Recommender")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["Recommendations", "Batch Compare", "Data Explorer", "Model Info"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**LightGCN** with attention-based layer aggregation "
    "on a user-item bipartite graph."
)

# ---------------------------------------------------------------------------
# Load everything once
# ---------------------------------------------------------------------------
config = load_config()
device = get_device()

try:
    interactions, num_users, num_items, edge_index = load_data(config)
    data_loaded = True
except Exception as e:
    data_loaded = False
    st.error(f"Failed to load data: {e}")

model_obj = None
model_meta = {}
if data_loaded:
    result = load_trained_model(config, num_users, num_items, device)
    if isinstance(result[0], str) or result[0] is None:
        model_loaded = False
        model_error = result[1] if isinstance(result[1], str) else "Unknown error"
    else:
        model_obj, model_meta = result
        model_loaded = True

# ============================================================================
# PAGE: Recommendations
# ============================================================================
if page == "Recommendations":
    st.title("Get Personalized Recommendations")

    if not data_loaded or not model_loaded:
        st.warning("Model or data not available. Please train the model first.")
        st.stop()

    col_input, col_results = st.columns([1, 2])

    with col_input:
        st.subheader("Input")
        user_id = st.number_input(
            "User ID",
            min_value=0,
            max_value=num_users - 1,
            value=0,
            step=1,
            help=f"Enter a user ID between 0 and {num_users - 1}",
        )
        top_k = st.slider(
            "Number of Recommendations",
            min_value=1,
            max_value=min(50, num_items),
            value=10,
        )
        run = st.button("Get Recommendations", type="primary", use_container_width=True)

        # Show user history
        st.markdown("---")
        st.subheader("User History")
        history = get_user_history(interactions, user_id)
        if len(history) > 0:
            hist_df = pd.DataFrame(history, columns=["userID", "itemID", "rating"])
            st.metric("Past Interactions", len(hist_df))
            st.metric("Avg Rating Given", f"{hist_df['rating'].mean():.2f}")
            st.dataframe(
                hist_df[["itemID", "rating"]].rename(
                    columns={"itemID": "Item ID", "rating": "Rating"}
                ),
                use_container_width=True,
                height=200,
            )
        else:
            st.info("No interaction history found for this user.")

    with col_results:
        if run:
            st.subheader(f"Top-{top_k} Recommendations for User {user_id}")
            with st.spinner("Generating recommendations..."):
                items, scores = get_recommendations(
                    model_obj, edge_index, user_id, top_k, device
                )

            rec_df = pd.DataFrame({
                "Rank": range(1, len(items) + 1),
                "Item ID": items,
                "Score": [f"{s:.4f}" for s in scores],
            })
            st.dataframe(rec_df, use_container_width=True, hide_index=True)

            # Bar chart of scores
            fig, ax = plt.subplots(figsize=(10, max(4, top_k * 0.3)))
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(items)))
            bars = ax.barh(
                [f"Item {i}" for i in reversed(items)],
                list(reversed(scores)),
                color=list(reversed(colors)),
            )
            ax.set_xlabel("Relevance Score")
            ax.set_title(f"Recommendation Scores — User {user_id}")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Configure inputs on the left and click **Get Recommendations**.")

# ============================================================================
# PAGE: Batch Compare
# ============================================================================
elif page == "Batch Compare":
    st.title("Compare Recommendations Across Users")

    if not data_loaded or not model_loaded:
        st.warning("Model or data not available.")
        st.stop()

    st.markdown("Enter multiple User IDs to compare their top recommendations side by side.")

    col_cfg, _ = st.columns([1, 2])
    with col_cfg:
        user_ids_text = st.text_input(
            "User IDs (comma-separated)",
            value="0, 1, 2",
            help="e.g. 0, 5, 42",
        )
        batch_top_k = st.slider("Top K", 1, 20, 10, key="batch_k")

    run_batch = st.button("Compare", type="primary")

    if run_batch:
        try:
            user_ids = [int(x.strip()) for x in user_ids_text.split(",") if x.strip()]
        except ValueError:
            st.error("Please enter valid integer User IDs separated by commas.")
            st.stop()

        invalid = [u for u in user_ids if u < 0 or u >= num_users]
        if invalid:
            st.error(f"Invalid User IDs: {invalid}. Must be in [0, {num_users}).")
            st.stop()

        cols = st.columns(len(user_ids))
        for idx, uid in enumerate(user_ids):
            with cols[idx]:
                items, scores = get_recommendations(
                    model_obj, edge_index, uid, batch_top_k, device
                )
                st.subheader(f"User {uid}")
                hist_count = len(get_user_history(interactions, uid))
                st.caption(f"{hist_count} past interactions")
                rec_df = pd.DataFrame({
                    "Rank": range(1, len(items) + 1),
                    "Item": items,
                    "Score": [round(s, 4) for s in scores],
                })
                st.dataframe(rec_df, use_container_width=True, hide_index=True)

        # Overlap analysis
        st.markdown("---")
        st.subheader("Overlap Analysis")
        all_recs = {}
        for uid in user_ids:
            items, _ = get_recommendations(
                model_obj, edge_index, uid, batch_top_k, device
            )
            all_recs[uid] = set(items)

        overlap_data = []
        for i, u1 in enumerate(user_ids):
            row = []
            for j, u2 in enumerate(user_ids):
                common = len(all_recs[u1] & all_recs[u2])
                row.append(common)
            overlap_data.append(row)

        overlap_df = pd.DataFrame(
            overlap_data,
            index=[f"User {u}" for u in user_ids],
            columns=[f"User {u}" for u in user_ids],
        )

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            overlap_df, annot=True, fmt="d", cmap="YlOrRd",
            ax=ax, linewidths=0.5,
        )
        ax.set_title(f"Shared Items in Top-{batch_top_k}")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ============================================================================
# PAGE: Data Explorer
# ============================================================================
elif page == "Data Explorer":
    st.title("Dataset Explorer & EDA")

    if not data_loaded:
        st.warning("Data not available.")
        st.stop()

    # Summary metrics
    st.subheader("Dataset Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Users", f"{num_users:,}")
    m2.metric("Items", f"{num_items:,}")
    m3.metric("Interactions", f"{len(interactions):,}")
    density = len(interactions) / (num_users * num_items) * 100
    m4.metric("Density", f"{density:.4f}%")

    st.markdown("---")

    # Tabs for different EDA views
    tab_ratings, tab_users, tab_items, tab_raw = st.tabs(
        ["Ratings Distribution", "User Activity", "Item Popularity", "Raw Data"]
    )

    try:
        df_raw = load_raw_dataframe(config)
    except Exception:
        df_raw = None

    with tab_ratings:
        if df_raw is not None:
            st.subheader("Ratings Distribution")
            fig, ax = plt.subplots(figsize=(8, 5))
            rating_counts = df_raw['overall'].value_counts().sort_index()
            bars = ax.bar(
                rating_counts.index.astype(str),
                rating_counts.values,
                color=sns.color_palette("viridis", len(rating_counts)),
                edgecolor="black",
            )
            for bar in bars:
                ax.annotate(
                    f'{int(bar.get_height()):,}',
                    (bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
                    ha='center', va='bottom', fontsize=10,
                )
            ax.set_xlabel("Rating")
            ax.set_ylabel("Count")
            ax.set_title("Distribution of Ratings")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            c1, c2, c3 = st.columns(3)
            c1.metric("Mean Rating", f"{df_raw['overall'].mean():.2f}")
            c2.metric("Median Rating", f"{df_raw['overall'].median():.1f}")
            c3.metric("Std Dev", f"{df_raw['overall'].std():.2f}")
        else:
            st.info("Raw data not available for EDA plots.")

    with tab_users:
        if df_raw is not None:
            st.subheader("User Activity")
            user_counts = df_raw['reviewerID'].value_counts()

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            axes[0].hist(user_counts.values, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
            axes[0].set_title("Interactions per User")
            axes[0].set_xlabel("Number of Interactions")
            axes[0].set_ylabel("Number of Users (log)")
            axes[0].set_yscale('log')

            sorted_c = sorted(user_counts.values)
            cdf = [i / len(sorted_c) for i in range(1, len(sorted_c) + 1)]
            axes[1].plot(sorted_c, cdf, color='steelblue', linewidth=2)
            axes[1].set_title("User Activity CDF")
            axes[1].set_xlabel("Number of Interactions")
            axes[1].set_ylabel("Cumulative Fraction")
            axes[1].grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            c1, c2, c3 = st.columns(3)
            c1.metric("Avg Interactions / User", f"{user_counts.mean():.1f}")
            c2.metric("Max", f"{user_counts.max():,}")
            c3.metric("Min", f"{user_counts.min()}")

    with tab_items:
        if df_raw is not None:
            st.subheader("Item Popularity")
            item_counts = df_raw['asin'].value_counts()

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            axes[0].hist(item_counts.values, bins=50, color='coral', edgecolor='black', alpha=0.7)
            axes[0].set_title("Interactions per Item")
            axes[0].set_xlabel("Number of Interactions")
            axes[0].set_ylabel("Number of Items (log)")
            axes[0].set_yscale('log')

            top20 = item_counts.head(20)
            axes[1].barh(
                [f"Item {i+1}" for i in range(len(top20))],
                top20.values,
                color='coral', alpha=0.7,
            )
            axes[1].set_title("Top 20 Most Popular Items")
            axes[1].set_xlabel("Interactions")
            axes[1].invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            c1, c2, c3 = st.columns(3)
            c1.metric("Avg Interactions / Item", f"{item_counts.mean():.1f}")
            c2.metric("Max", f"{item_counts.max():,}")
            c3.metric("Min", f"{item_counts.min()}")

    with tab_raw:
        st.subheader("Raw Data Preview")
        if df_raw is not None:
            st.dataframe(df_raw.head(500), use_container_width=True, height=400)
            st.caption(f"Showing first 500 of {len(df_raw):,} rows")
        else:
            st.info("Raw dataframe not available.")

# ============================================================================
# PAGE: Model Info
# ============================================================================
elif page == "Model Info":
    st.title("Model Information & Health")

    # Config details
    st.subheader("Configuration")
    cfg_col1, cfg_col2, cfg_col3 = st.columns(3)
    with cfg_col1:
        st.markdown("**Model**")
        st.json({
            "embedding_dim": config.model.embedding_dim,
            "num_layers": config.model.num_layers,
            "dropout": config.model.dropout,
            "use_attention": config.model.use_attention,
        })
    with cfg_col2:
        st.markdown("**Training**")
        st.json({
            "epochs": config.training.epochs,
            "learning_rate": config.training.learning_rate,
            "batch_size": config.training.batch_size,
            "weight_decay": config.training.weight_decay,
            "early_stopping_patience": config.training.early_stopping_patience,
        })
    with cfg_col3:
        st.markdown("**Evaluation**")
        st.json({
            "top_k_values": config.evaluation.top_k_values,
            "metrics": config.evaluation.metrics,
        })

    st.markdown("---")

    # Model architecture
    st.subheader("Model Architecture")
    if data_loaded and model_loaded:
        arch_col1, arch_col2 = st.columns(2)
        with arch_col1:
            st.markdown("**LightGCN with Attention**")
            st.markdown(f"""
| Component | Details |
|---|---|
| Users | {num_users:,} |
| Items | {num_items:,} |
| Embedding Dim | {config.model.embedding_dim} |
| GCN Layers | {config.model.num_layers} |
| Aggregation | Mean (scatter_mean) |
| Layer Fusion | Learned attention weights |
| Loss | BPR + L2 regularization |
| Optimizer | Adam |
""")
        with arch_col2:
            st.markdown("**Parameter Count**")
            total_params = sum(p.numel() for p in model_obj.parameters())
            trainable = sum(p.numel() for p in model_obj.parameters() if p.requires_grad)
            st.metric("Total Parameters", f"{total_params:,}")
            st.metric("Trainable Parameters", f"{trainable:,}")

            if model_meta:
                st.markdown("**Checkpoint Metadata**")
                st.json(model_meta)
    else:
        st.warning(
            "Model not loaded. Train with `python main.py --mode train` first."
        )

    st.markdown("---")

    # Device info
    st.subheader("System Info")
    sys_c1, sys_c2 = st.columns(2)
    with sys_c1:
        st.markdown(f"**Device:** `{device}`")
        st.markdown(f"**PyTorch:** `{torch.__version__}`")
        if torch.cuda.is_available():
            st.markdown(f"**GPU:** `{torch.cuda.get_device_name(0)}`")
            mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            st.markdown(f"**GPU Memory:** `{mem:.1f} GB`")
    with sys_c2:
        st.markdown(f"**Data path:** `{config.get_data_path()}`")
        st.markdown(f"**Model path:** `{config.get_model_save_path()}`")
        model_file = config.get_model_save_path()
        if model_file.exists():
            size_mb = model_file.stat().st_size / (1024 * 1024)
            st.markdown(f"**Model file size:** `{size_mb:.2f} MB`")
