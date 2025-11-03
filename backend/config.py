"""
Configuration module for the Real-time System Monitoring application.

This module defines configuration classes and constants for the application,
including paths, limits, and performance tuning parameters.
"""
import os
from pathlib import Path


class Config:
    """Base configuration class with default settings."""
    
    # Security
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-change-in-production'
    DEBUG = os.environ.get('DEBUG', 'True').lower() in ('true', '1', 't')
    
    # Database configuration
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///system_metrics.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ECHO = False  # Set to True for SQL query debugging
    
    # Paths
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR.parent / 'data'
    RAW_DATA_DIR = DATA_DIR / 'raw'
    UPLOAD_FOLDER = str(RAW_DATA_DIR)
    MODEL_FOLDER = str(BASE_DIR.parent / 'ml' / 'models')
    LOG_DIR = BASE_DIR.parent / 'logs'
    LOG_FILE = str(LOG_DIR / 'app.log')
    
    # Upload configuration
    ALLOWED_EXTENSIONS = {'csv', 'log', 'txt'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max file size
    
    # Metrics collection
    METRICS_COLLECTION_INTERVAL = 60  # seconds between metrics writes
    METRICS_HISTORY_LIMIT = 10000  # Maximum number of records to keep in memory
    
    # ML Model configuration
    ANOMALY_CONTAMINATION = 0.05  # Expected proportion of anomalies
    FORECAST_PERIODS = 24  # Number of periods to forecast
    FORECAST_FREQUENCY = 'H'  # Hourly frequency
    CLUSTERING_N_CLUSTERS = 3  # Default number of clusters
    
    # Performance tuning
    PROCESS_ITER_CACHE_TIMEOUT = 5  # Seconds to cache process iteration
    NETWORK_SPEED_CACHE_TIMEOUT = 1  # Seconds to cache network speed calculations
    
    # Visualization
    PLOT_DPI = 100  # DPI for saved plots
    PLOT_STYLE = 'seaborn-v0_8-darkgrid'  # Matplotlib style
    
    @classmethod
    def init_app(cls):
        """Initialize application directories."""
        # Create necessary directories
        for directory in [cls.DATA_DIR, cls.RAW_DATA_DIR, cls.LOG_DIR]:
            directory.mkdir(parents=True, exist_ok=True)


class DevelopmentConfig(Config):
    """Development configuration with debugging enabled."""
    DEBUG = True
    SQLALCHEMY_ECHO = True


class ProductionConfig(Config):
    """Production configuration with security hardening."""
    DEBUG = False
    SQLALCHEMY_ECHO = False
    
    # Override with stronger security requirements
    @classmethod
    def init_app(cls):
        Config.init_app()
        
        # Ensure SECRET_KEY is set in production
        if cls.SECRET_KEY == 'dev-key-change-in-production':
            raise ValueError("SECRET_KEY must be set in production environment")


class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}