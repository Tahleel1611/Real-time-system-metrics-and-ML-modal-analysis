"""
Time series forecasting module for system metrics.

This module provides a SystemMetricsForecaster class for predicting
future values of system metrics using the Prophet forecasting library.
"""
import pandas as pd
from prophet import Prophet
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Optional
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)


class SystemMetricsForecaster:
    """
    Time series forecaster for system metrics using Facebook Prophet.
    
    Attributes:
        periods: Number of periods to forecast
        frequency: Frequency of the forecast ('H' for hourly, 'D' for daily, etc.)
        models: Dictionary storing trained Prophet models for each metric
    """
    
    def __init__(self, periods: int = 24, frequency: str = 'H'):
        """
        Initialize the forecaster with default parameters.
        
        Args:
            periods: Number of periods to forecast (default: 24)
            frequency: Frequency of the forecast (default: 'H' for hourly)
        """
        if periods < 1:
            raise ValueError("periods must be at least 1")
        
        valid_frequencies = ['S', 'T', 'H', 'D', 'W', 'M', 'Y']
        if frequency not in valid_frequencies:
            raise ValueError(f"frequency must be one of {valid_frequencies}")
        
        self.periods = periods
        self.frequency = frequency
        self.models = {}
        
    def prepare_data(self, data: pd.DataFrame, metric: str) -> pd.DataFrame:
        """
        Prepare data for Prophet model.
        
        Args:
            data: pandas DataFrame containing the data
            metric: Column name of the metric to forecast
            
        Returns:
            pandas DataFrame in Prophet format with 'ds' and 'y' columns
            
        Raises:
            ValueError: If metric column doesn't exist or data is invalid
        """
        if metric not in data.columns:
            raise ValueError(f"Metric '{metric}' not found in data")
        
        df = data.reset_index()
        
        # Try to find timestamp column
        timestamp_col = None
        for col in ['timestamp', 'Timestamp', 'ds', 'date', 'Date']:
            if col in df.columns:
                timestamp_col = col
                break
        
        if timestamp_col is None:
            # Use index if no timestamp column found
            df['ds'] = df.index
        else:
            df = df.rename(columns={timestamp_col: 'ds'})
        
        df = df.rename(columns={metric: 'y'})
        df['ds'] = pd.to_datetime(df['ds'])
        
        # Remove any rows with missing values
        df = df[['ds', 'y']].dropna()
        
        if len(df) < 2:
            raise ValueError(f"Insufficient data for forecasting. Need at least 2 points, got {len(df)}")
        
        return df
    
    def train(self, data: pd.DataFrame, metrics: Optional[List[str]] = None) -> None:
        """
        Train Prophet models for specified metrics.
        
        Args:
            data: pandas DataFrame containing the training data
            metrics: List of metric names to train. If None, uses default metrics.
            
        Raises:
            ValueError: If no valid metrics are found in data
        """
        # Default metrics to forecast
        if metrics is None:
            metrics = ['cpu', 'gpu', 'memory_usage', 'data_in', 'data_out']
        
        # Filter to only metrics that exist in the data
        available_metrics = [m for m in metrics if m in data.columns]
        
        if not available_metrics:
            raise ValueError(f"None of the specified metrics {metrics} found in data")
        
        print(f"Training forecasting models for: {', '.join(available_metrics)}")
        
        for metric in available_metrics:
            try:
                metric_data = self.prepare_data(data, metric)
                
                # Configure Prophet with appropriate seasonality
                model = Prophet(
                    yearly_seasonality=False,  # Typically not needed for system metrics
                    weekly_seasonality=True,
                    daily_seasonality=True,
                    seasonality_mode='additive',
                    interval_width=0.95
                )
                
                # Suppress Prophet's verbose output
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(metric_data)
                
                self.models[metric] = model
                print(f"  ✓ Trained model for {metric}")
                
            except Exception as e:
                print(f"  ✗ Failed to train model for {metric}: {e}")
                continue
        
        if not self.models:
            raise ValueError("Failed to train any models")
    
    def forecast(self, metric: str) -> pd.DataFrame:
        """
        Generate forecast for a specific metric.
        
        Args:
            metric: The metric to forecast (e.g., 'cpu', 'gpu')
            
        Returns:
            pandas DataFrame with forecast results (ds, yhat, yhat_lower, yhat_upper)
            
        Raises:
            ValueError: If no model has been trained for the metric
        """
        if metric not in self.models:
            available = list(self.models.keys())
            raise ValueError(f"No model trained for metric '{metric}'. Available: {available}")
            
        model = self.models[metric]
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=self.periods, freq=self.frequency)
        
        # Generate forecast
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast = model.predict(future)
        
        # Return only relevant columns
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    def get_all_forecasts(self) -> Dict[str, pd.DataFrame]:
        """
        Generate forecasts for all trained metrics.
        
        Returns:
            Dictionary mapping metric names to forecast DataFrames
        """
        forecasts = {}
        for metric in self.models.keys():
            try:
                forecasts[metric] = self.forecast(metric)
            except Exception as e:
                print(f"Warning: Failed to generate forecast for {metric}: {e}")
                continue
        return forecasts
    
    def is_trained(self, metric: str) -> bool:
        """
        Check if a model has been trained for a specific metric.
        
        Args:
            metric: Metric name to check
            
        Returns:
            True if model exists, False otherwise
        """
        return metric in self.models

def create_forecast(data: pd.DataFrame, periods: int = 24, frequency: str = 'H',
                   metrics: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    """
    Main function to create forecasts for system metrics.
    
    Args:
        data: pandas DataFrame containing system metrics
        periods: Number of periods to forecast (default: 24)
        frequency: Forecast frequency (default: 'H' for hourly)
        metrics: List of metrics to forecast. If None, uses all available metrics.
    
    Returns:
        Dictionary mapping metric names to forecast DataFrames
        
    Raises:
        ValueError: If data is invalid or insufficient
    """
    if len(data) < 10:
        raise ValueError(f"Insufficient data for forecasting. Need at least 10 samples, got {len(data)}")
    
    # Initialize the forecaster
    forecaster = SystemMetricsForecaster(periods=periods, frequency=frequency)
    
    # Train the models
    print(f"Training forecasting models (periods={periods}, frequency={frequency})...")
    forecaster.train(data, metrics=metrics)
    
    # Get forecasts for all metrics
    print("Generating forecasts...")
    forecasts = forecaster.get_all_forecasts()
    
    print(f"Successfully generated forecasts for {len(forecasts)} metrics")
    return forecasts


if __name__ == "__main__":
    """Example usage of the forecasting module."""
    
    # Generate sample data for demonstration
    print("Generating sample system metrics data...")
    n_samples = 168  # 1 week of hourly data
    dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='H')
    
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'cpu': 50 + 10 * np.sin(np.arange(n_samples) * 2 * np.pi / 24) + np.random.normal(0, 5, n_samples),
        'gpu': 60 + 15 * np.sin(np.arange(n_samples) * 2 * np.pi / 24) + np.random.normal(0, 8, n_samples),
        'memory_usage': 70 + 5 * np.sin(np.arange(n_samples) * 2 * np.pi / 24) + np.random.normal(0, 3, n_samples),
        'data_in': 1000 + 200 * np.sin(np.arange(n_samples) * 2 * np.pi / 24) + np.random.normal(0, 100, n_samples),
        'data_out': 500 + 100 * np.sin(np.arange(n_samples) * 2 * np.pi / 24) + np.random.normal(0, 50, n_samples)
    })
    
    # Ensure non-negative values
    for col in ['cpu', 'gpu', 'memory_usage', 'data_in', 'data_out']:
        sample_data[col] = sample_data[col].clip(lower=0, upper=100 if col in ['cpu', 'gpu', 'memory_usage'] else None)
    
    sample_data.set_index('timestamp', inplace=True)
    
    print(f"\nSample data shape: {sample_data.shape}")
    print(f"Date range: {sample_data.index[0]} to {sample_data.index[-1]}")
    
    # Create forecasts
    try:
        forecasts = create_forecast(sample_data, periods=24, frequency='H')
        
        print("\n" + "="*80)
        print("FORECAST RESULTS")
        print("="*80)
        
        for metric, forecast_df in forecasts.items():
            print(f"\n{metric.upper()}:")
            print(f"  Last actual value: {sample_data[metric].iloc[-1]:.2f}")
            print(f"  24h forecast (mean): {forecast_df['yhat'].tail(24).mean():.2f}")
            print(f"  24h forecast range: [{forecast_df['yhat_lower'].tail(24).mean():.2f}, "
                  f"{forecast_df['yhat_upper'].tail(24).mean():.2f}]")
        
    except Exception as e:
        print(f"Error during forecasting: {e}")
        import traceback
        traceback.print_exc()