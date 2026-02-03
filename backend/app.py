"""
FastAPI Application for LightGCN Recommendation System

Production-ready REST API with health checks, CORS, input validation,
proper error handling, and thread-safe model serving.
"""

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.config_manager import get_config, setup_logging, ConfigManager
from backend.model import LightGCNAttention
from backend.data_loader import load_amazon_reviews, build_edge_index
from backend.utils import load_model, get_device

logger = logging.getLogger(__name__)


# --- Application State ---
class AppState:
    """Thread-safe application state container."""

    def __init__(self):
        self.model: Optional[LightGCNAttention] = None
        self.edge_index: Optional[torch.Tensor] = None
        self.num_users: int = 0
        self.num_items: int = 0
        self.device: torch.device = torch.device('cpu')
        self.is_ready: bool = False
        self.model_load_time: Optional[float] = None

    def load(self, config=None):
        """Load model and data for serving."""
        if config is None:
            config = get_config()

        start_time = time.time()
        logger.info("Loading model and data for serving...")

        # Load data
        data_path = str(config.get_data_path())
        cache_dir = str(config.get_cache_dir())
        interactions, self.num_users, self.num_items = load_amazon_reviews(
            data_path, cache_dir=cache_dir
        )
        self.edge_index = build_edge_index(interactions, self.num_users)

        # Load model
        model_path = str(config.get_model_save_path())
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model file not found at {model_path}. "
                f"Train the model first with: python main.py --mode train"
            )

        self.device = get_device()

        try:
            # Try loading new checkpoint format
            self.model, metadata = load_model(model_path, self.device)
            logger.info(f"Loaded model checkpoint with metadata: {metadata}")
        except (KeyError, TypeError):
            # Fall back to legacy state_dict format
            logger.info("Loading legacy model format (state_dict only)")
            self.model = LightGCNAttention(
                self.num_users, self.num_items,
                config.model.embedding_dim,
                config.model.num_layers
            )
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device, weights_only=False)
            )
            self.model.to(self.device)
            self.model.eval()

        self.edge_index = self.edge_index.to(self.device)
        self.model_load_time = time.time() - start_time
        self.is_ready = True

        logger.info(
            f"Model loaded in {self.model_load_time:.2f}s | "
            f"Users: {self.num_users} | Items: {self.num_items} | "
            f"Device: {self.device}"
        )


app_state = AppState()


# --- Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    # Reset singleton to allow fresh config loading
    ConfigManager._instance = None
    ConfigManager._config = None

    config = get_config()
    setup_logging(config)
    logger.info("Starting LightGCN Recommendation API...")

    try:
        app_state.load(config)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.warning("API starting without model - only health check available")

    yield

    logger.info("Shutting down LightGCN Recommendation API...")


# --- FastAPI App ---
app = FastAPI(
    title="LightGCN Recommendation API",
    description="Production-ready GNN-based recommendation system using LightGCN with attention",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request/Response Models ---
class UserRequest(BaseModel):
    """Recommendation request schema."""
    user_id: int = Field(..., ge=0, description="User ID to get recommendations for")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of recommendations")


class RecommendationResponse(BaseModel):
    """Recommendation response schema."""
    user_id: int
    recommended_items: List[int]
    num_recommendations: int


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str
    model_loaded: bool
    num_users: int
    num_items: int
    device: str
    model_load_time: Optional[float]


class BatchRequest(BaseModel):
    """Batch recommendation request schema."""
    user_ids: List[int] = Field(..., min_length=1, max_length=100)
    top_k: int = Field(default=10, ge=1, le=100)


class BatchRecommendationResponse(BaseModel):
    """Batch recommendation response schema."""
    recommendations: List[RecommendationResponse]


# --- Endpoints ---
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for monitoring and load balancers."""
    return HealthResponse(
        status="healthy" if app_state.is_ready else "degraded",
        model_loaded=app_state.is_ready,
        num_users=app_state.num_users,
        num_items=app_state.num_items,
        device=str(app_state.device),
        model_load_time=app_state.model_load_time,
    )


@app.post("/recommend/", response_model=RecommendationResponse)
async def recommend(req: UserRequest):
    """
    Get top-K recommendations for a single user.

    Args:
        req: UserRequest with user_id and top_k.

    Returns:
        RecommendationResponse with recommended item IDs.
    """
    if not app_state.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service is starting up."
        )

    if req.user_id >= app_state.num_users:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid user_id: {req.user_id}. Must be < {app_state.num_users}"
        )

    top_k = min(req.top_k, app_state.num_items)

    try:
        recommended = app_state.model.predict(
            req.user_id, app_state.edge_index, top_k
        )
        items = recommended.cpu().tolist()
    except Exception as e:
        logger.error(f"Recommendation failed for user {req.user_id}: {e}")
        raise HTTPException(status_code=500, detail="Recommendation generation failed")

    return RecommendationResponse(
        user_id=req.user_id,
        recommended_items=items,
        num_recommendations=len(items),
    )


@app.post("/recommend/batch", response_model=BatchRecommendationResponse)
async def recommend_batch(req: BatchRequest):
    """
    Get top-K recommendations for multiple users.

    Args:
        req: BatchRequest with list of user_ids and top_k.

    Returns:
        BatchRecommendationResponse with recommendations for each user.
    """
    if not app_state.is_ready:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate all user IDs
    invalid_ids = [uid for uid in req.user_ids if uid >= app_state.num_users or uid < 0]
    if invalid_ids:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid user_ids: {invalid_ids}. Must be in [0, {app_state.num_users})"
        )

    top_k = min(req.top_k, app_state.num_items)

    try:
        app_state.model.eval()
        with torch.no_grad():
            user_emb, item_emb = app_state.model(app_state.edge_index)
            recommendations = []
            for user_id in req.user_ids:
                scores = torch.matmul(user_emb[user_id], item_emb.T)
                _, top_items = torch.topk(scores, top_k)
                items = top_items.cpu().tolist()
                recommendations.append(RecommendationResponse(
                    user_id=user_id,
                    recommended_items=items,
                    num_recommendations=len(items),
                ))
    except Exception as e:
        logger.error(f"Batch recommendation failed: {e}")
        raise HTTPException(status_code=500, detail="Batch recommendation failed")

    return BatchRecommendationResponse(recommendations=recommendations)


# --- Serve frontend static files ---
_frontend_dir = Path(__file__).parent.parent / "frontend"
if _frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_frontend_dir / "static")), name="static")

    @app.get("/")
    async def serve_frontend():
        """Serve the frontend application."""
        index_path = _frontend_dir / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))
        raise HTTPException(status_code=404, detail="Frontend not found")
