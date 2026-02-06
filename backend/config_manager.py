"""
Configuration Manager for LightGCN Recommendation System

Handles configuration loading from YAML and environment variables,
provides path resolution and validation.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
from functools import lru_cache

logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.resolve()


@dataclass
class DataConfig:
    """Data configuration settings."""
    file_path: str
    cache_dir: str = "cache"
    min_interactions: int = 5


@dataclass
class ModelConfig:
    """Model configuration settings."""
    embedding_dim: int = 32
    num_layers: int = 3
    dropout: float = 0.1
    use_attention: bool = True


@dataclass
class TrainingConfig:
    """Training configuration settings."""
    epochs: int = 10
    batch_size: int = 1024
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    early_stopping_patience: int = 3
    validation_split: float = 0.1
    test_split: float = 0.1
    checkpoint_dir: str = "checkpoints"
    model_save_path: str = "saved_models/lightgcn_model.pth"


@dataclass
class EvaluationConfig:
    """Evaluation configuration settings."""
    top_k_values: list = field(default_factory=lambda: [5, 10, 20])
    metrics: list = field(default_factory=lambda: ['hit_ratio', 'ndcg', 'precision', 'recall', 'mrr'])


@dataclass
class APIConfig:
    """API configuration settings."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    reload: bool = False
    cors_origins: list = field(default_factory=lambda: ['*'])
    max_recommendations: int = 100
    request_timeout: int = 30


@dataclass
class LoggingConfig:
    """Logging configuration settings."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "logs/app.log"
    max_bytes: int = 10485760
    backup_count: int = 5


@dataclass
class EDAConfig:
    """EDA configuration settings."""
    output_dir: str = "reports"
    plots: list = field(default_factory=lambda: [
        'ratings_distribution', 'user_activity', 'item_popularity', 'interaction_timeline'
    ])


@dataclass
class Config:
    """Main configuration class containing all settings."""
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    api: APIConfig
    logging: LoggingConfig
    eda: EDAConfig
    project_root: Path = field(default_factory=get_project_root)

    def resolve_path(self, path: str) -> Path:
        """Resolve a relative path to absolute using project root."""
        path_obj = Path(path)
        if path_obj.is_absolute():
            return path_obj
        return self.project_root / path

    def get_data_path(self) -> Path:
        """Get the absolute path to the data file."""
        return self.resolve_path(self.data.file_path)

    def get_model_save_path(self) -> Path:
        """Get the absolute path for model saving."""
        return self.resolve_path(self.training.model_save_path)

    def get_checkpoint_dir(self) -> Path:
        """Get the absolute path for checkpoints."""
        return self.resolve_path(self.training.checkpoint_dir)

    def get_cache_dir(self) -> Path:
        """Get the absolute path for cache."""
        return self.resolve_path(self.data.cache_dir)

    def get_log_file_path(self) -> Path:
        """Get the absolute path for log file."""
        return self.resolve_path(self.logging.file)

    def get_eda_output_dir(self) -> Path:
        """Get the absolute path for EDA outputs."""
        return self.resolve_path(self.eda.output_dir)

    def ensure_directories(self) -> None:
        """Create all necessary directories if they don't exist."""
        directories = [
            self.get_checkpoint_dir(),
            self.get_cache_dir(),
            self.get_log_file_path().parent,
            self.get_eda_output_dir(),
            self.get_model_save_path().parent,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")


class ConfigManager:
    """Manages configuration loading and validation."""

    _instance: Optional['ConfigManager'] = None
    _config: Optional[Config] = None

    def __new__(cls) -> 'ConfigManager':
        """Singleton pattern to ensure single configuration instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_config(self, config_path: Optional[str] = None) -> Config:
        """
        Load configuration from YAML file with environment variable overrides.

        Args:
            config_path: Path to config file. If None, uses default location.

        Returns:
            Config object with all settings.
        """
        if self._config is not None:
            return self._config

        # Determine config path
        if config_path is None:
            config_path = os.environ.get(
                'LIGHTGCN_CONFIG_PATH',
                str(get_project_root() / 'backend' / 'config.yaml')
            )

        config_path = Path(config_path)
        if not config_path.exists():
            logger.warning(f"Config file not found at {config_path}, using defaults")
            yaml_config = {}
        else:
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f) or {}

        # Apply environment variable overrides
        yaml_config = self._apply_env_overrides(yaml_config)

        # Build configuration objects
        self._config = Config(
            data=DataConfig(**yaml_config.get('data', {})),
            model=ModelConfig(**yaml_config.get('model', {})),
            training=TrainingConfig(**yaml_config.get('training', {})),
            evaluation=EvaluationConfig(**yaml_config.get('evaluation', {})),
            api=APIConfig(**yaml_config.get('api', {})),
            logging=LoggingConfig(**yaml_config.get('logging', {})),
            eda=EDAConfig(**yaml_config.get('eda', {})),
        )

        # Ensure directories exist
        self._config.ensure_directories()

        return self._config

    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides to configuration."""
        env_mappings = {
            'LIGHTGCN_DATA_PATH': ('data', 'file_path'),
            'LIGHTGCN_EMBEDDING_DIM': ('model', 'embedding_dim', int),
            'LIGHTGCN_NUM_LAYERS': ('model', 'num_layers', int),
            'LIGHTGCN_EPOCHS': ('training', 'epochs', int),
            'LIGHTGCN_LEARNING_RATE': ('training', 'learning_rate', float),
            'LIGHTGCN_BATCH_SIZE': ('training', 'batch_size', int),
            'LIGHTGCN_API_HOST': ('api', 'host'),
            'LIGHTGCN_API_PORT': ('api', 'port', int),
            'LIGHTGCN_LOG_LEVEL': ('logging', 'level'),
        }

        for env_var, mapping in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                section = mapping[0]
                key = mapping[1]
                converter = mapping[2] if len(mapping) > 2 else str

                if section not in config:
                    config[section] = {}

                try:
                    config[section][key] = converter(value)
                    logger.info(f"Applied env override: {env_var}={value}")
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to apply env override {env_var}: {e}")

        return config

    def reset(self) -> None:
        """Reset configuration (useful for testing)."""
        self._config = None

    @property
    def config(self) -> Config:
        """Get the current configuration, loading if necessary."""
        if self._config is None:
            return self.load_config()
        return self._config


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Get the singleton configuration instance."""
    return ConfigManager().load_config()


def setup_logging(config: Optional[Config] = None) -> None:
    """
    Set up logging based on configuration.

    Args:
        config: Configuration object. If None, loads from default.
    """
    if config is None:
        config = get_config()

    log_config = config.logging

    # Create log directory
    log_path = config.get_log_file_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_config.level.upper()))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_config.level.upper()))
    console_formatter = logging.Formatter(log_config.format)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler with rotation
    try:
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=log_config.max_bytes,
            backupCount=log_config.backup_count
        )
        file_handler.setLevel(getattr(logging, log_config.level.upper()))
        file_formatter = logging.Formatter(log_config.format)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"Could not set up file logging: {e}")

    logger.info("Logging configured successfully")
