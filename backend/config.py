"""
Configuration Module
Centralized config management using Pydantic + environment variables.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings from .env file."""

    # Database
    database_url: str = "sqlite:///fraud_detection.db"
    db_echo: bool = False

    # API Security
    api_key_secret: str = "change-me-in-production"
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # Model Paths
    model_dir: str = "models"
    data_dir: str = "data"
    processed_data_dir: str = "data/processed"

    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None  # None = console only

    # API Server
    api_port: int = 8000
    api_host: str = "0.0.0.0"
    workers: int = 4
    reload: bool = False

    # Feature Store
    feature_cache_size: int = 10000
    feature_ttl_seconds: int = 3600

    # Monitoring
    enable_prometheus: bool = True
    prometheus_port: int = 9090

    # Model Drift Detection
    drift_check_interval_days: int = 7
    drift_threshold_kl_divergence: float = 0.15
    drift_threshold_js_divergence: float = 0.1

    # Environment
    debug: bool = False
    environment: str = "development"

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """
    Cached settings singleton.
    Use this in FastAPI dependencies.
    """
    return Settings()


if __name__ == "__main__":
    settings = get_settings()
    print(f"Database: {settings.database_url}")
    print(f"Log Level: {settings.log_level}")
