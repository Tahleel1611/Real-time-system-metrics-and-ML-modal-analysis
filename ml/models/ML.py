"""
Machine Learning module for system metrics analysis.

This module provides comprehensive ML analysis including:
- Anomaly detection using Isolation Forest
- Time series decomposition and forecasting
- Clustering analysis
- Visualization of results

Optimized for performance with proper error handling and efficient data processing.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from typing import Dict, Tuple, Any
import warnings

# Suppress Prophet warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)


def load_and_prepare_data(file_path: str) -> pd.DataFrame:
    """
    Load and prepare data from CSV file.
    
    Args:
        file_path: Path to the CSV file containing metrics data
        
    Returns:
        DataFrame with Timestamp as index
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If data format is invalid
    """
    try:
        df = pd.read_csv(file_path, parse_dates=['Timestamp'])
        df.set_index('Timestamp', inplace=True)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {file_path}")
    except Exception as e:
        raise ValueError(f"Error loading data: {e}")


def detect_anomalies(df: pd.DataFrame, contamination: float = 0.1) -> pd.DataFrame:
    """
    Detect anomalies in the dataset using Isolation Forest.
    
    Args:
        df: DataFrame containing metrics data
        contamination: Expected proportion of anomalies (default: 0.1)
        
    Returns:
        DataFrame with added 'anomaly' column (-1 for anomaly, 1 for normal)
    """
    # Create a copy to avoid modifying original
    df = df.copy()
    
    # Prepare data for anomaly detection
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    # Use Isolation Forest for anomaly detection
    iso_forest = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    anomalies = iso_forest.fit_predict(scaled_data)
    
    df['anomaly'] = anomalies
    return df

def analyze_time_series(df: pd.DataFrame, column: str, period: int = 24) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Perform time series decomposition.
    
    Args:
        df: DataFrame containing the time series data
        column: Column name to analyze
        period: Seasonal period (default: 24 for hourly data)
        
    Returns:
        Tuple of (trend, seasonal, residual) components
        
    Raises:
        ValueError: If column doesn't exist or data is insufficient
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if len(df) < 2 * period:
        raise ValueError(f"Insufficient data points. Need at least {2 * period}, got {len(df)}")
    
    # Perform time series decomposition
    decomposition = seasonal_decompose(df[column], model='additive', period=period, extrapolate_trend='freq')
    
    return decomposition.trend, decomposition.seasonal, decomposition.resid


def forecast_with_prophet(df: pd.DataFrame, column: str, periods: int = 24, freq: str = 'H') -> pd.DataFrame:
    """
    Generate forecast using Prophet.
    
    Args:
        df: DataFrame containing historical data
        column: Column name to forecast
        periods: Number of periods to forecast (default: 24)
        freq: Frequency string (default: 'H' for hourly)
        
    Returns:
        DataFrame with forecast results including yhat, yhat_lower, yhat_upper
    """
    # Prepare data for Prophet
    prophet_df = df.reset_index().rename(columns={'Timestamp': 'ds', column: 'y'})
    prophet_df = prophet_df[['ds', 'y']].dropna()

    if len(prophet_df) < 2:
        raise ValueError(f"Insufficient data for forecasting. Need at least 2 points, got {len(prophet_df)}")

    # Create and fit the model with optimized settings
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=False,
        yearly_seasonality=False,
        seasonality_mode='additive'
    )
    
    # Suppress Prophet's verbose output
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(prophet_df)

    # Make future dataframe for predictions
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    
    return forecast

def plot_results(df: pd.DataFrame, column: str, trend: pd.Series, seasonal: pd.Series, 
                residual: pd.Series, forecast: pd.DataFrame, save_path: str = None) -> None:
    """
    Create comprehensive visualization of analysis results.
    
    Args:
        df: DataFrame containing original data
        column: Column name being analyzed
        trend: Trend component from decomposition
        seasonal: Seasonal component from decomposition
        residual: Residual component from decomposition
        forecast: Forecast results from Prophet
        save_path: Optional path to save the figure instead of displaying
    """
    # Determine number of subplots (5 or 6 depending on column)
    n_plots = 6 if column == 'HBM_Usage' else 5
    fig, axes = plt.subplots(n_plots, 1, figsize=(15, 5 * n_plots))

    # Original data with anomalies
    df.plot(y=column, ax=axes[0], title=f'Original {column} Data with Anomalies', legend=False)
    if 'anomaly' in df.columns:
        anomalies = df[df['anomaly'] == -1]
        if len(anomalies) > 0:
            axes[0].scatter(anomalies.index, anomalies[column], color='red', label='Anomaly', s=50, zorder=5)
            axes[0].legend()

    # Trend
    trend.plot(ax=axes[1], title=f'{column} Trend', color='green')

    # Seasonal
    seasonal.plot(ax=axes[2], title=f'{column} Seasonal Pattern', color='orange')

    # Residual
    residual.plot(ax=axes[3], title=f'{column} Residual', color='purple')

    # Forecast
    axes[4].plot(df.index, df[column], label='Actual', color='blue')
    axes[4].plot(forecast['ds'], forecast['yhat'], color='red', label='Forecast', linestyle='--')
    axes[4].fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], 
                          alpha=0.3, color='red', label='Confidence Interval')
    axes[4].set_title(f'{column} Forecast')
    axes[4].legend()
    axes[4].set_xlabel('Time')
    axes[4].set_ylabel(column)

    # HBM usage (only for HBM_Usage column)
    if column == 'HBM_Usage':
        axes[5].plot(df.index, df[column], label='HBM Usage', color='teal')
        axes[5].set_title('HBM Usage Over Time')
        axes[5].set_xlabel('Time')
        axes[5].set_ylabel('HBM Usage (%)')
        axes[5].legend()

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
    else:
        plt.show(block=False)
        plt.pause(0.1)

def cluster_analysis(df: pd.DataFrame, n_clusters: int = 3, save_path: str = None) -> np.ndarray:
    """
    Perform clustering analysis on system metrics.
    
    Args:
        df: DataFrame containing metrics data
        n_clusters: Number of clusters (default: 3)
        save_path: Optional path to save the figure
        
    Returns:
        Array of cluster labels for each data point
    """
    # Prepare data for clustering (remove anomaly column if present)
    df_clean = df.drop('anomaly', axis=1) if 'anomaly' in df.columns else df.copy()
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_clean)

    # Perform K-means clustering with optimized parameters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_data)

    # Perform PCA for visualization
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)

    # Plot clusters
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=clusters, cmap='viridis', alpha=0.6, edgecolors='w', s=50)
    plt.title('Cluster Analysis of System Metrics', fontsize=14, fontweight='bold')
    plt.xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    plt.ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
    else:
        plt.show(block=False)
        plt.pause(0.1)

    return clusters

def analyze_logs(file_path: str, visualize: bool = True, save_plots: bool = False, 
                 output_dir: str = 'output') -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Perform comprehensive analysis on system log data.
    
    Args:
        file_path: Path to CSV file containing log data
        visualize: Whether to generate plots (default: True)
        save_plots: Whether to save plots to disk instead of displaying (default: False)
        output_dir: Directory to save plots if save_plots=True
        
    Returns:
        Tuple of (analyzed_dataframe, results_dictionary)
    """
    import os
    
    # Load data
    print(f"Loading data from {file_path}...")
    df = load_and_prepare_data(file_path)
    
    # Detect anomalies
    print("Detecting anomalies...")
    df = detect_anomalies(df)
    
    # Create output directory if saving plots
    if save_plots and visualize:
        os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    columns_to_analyze = ['CPU', 'GPU', 'Memory_Usage', 'HBM_Usage', 'Data_In', 'Data_Out']
    
    for column in columns_to_analyze:
        if column not in df.columns:
            print(f"Warning: Column '{column}' not found in data. Skipping...")
            continue
            
        print(f"Analyzing {column}...")
        
        try:
            # Time series analysis
            trend, seasonal, residual = analyze_time_series(df, column)
            
            # Forecasting
            forecast = forecast_with_prophet(df, column)
            
            # Plot results if requested
            if visualize:
                save_path = os.path.join(output_dir, f'{column}_analysis.png') if save_plots else None
                plot_results(df, column, trend, seasonal, residual, forecast, save_path)
            
            # Store results
            results[column] = {
                'trend': trend,
                'seasonal': seasonal,
                'residual': residual,
                'forecast': forecast
            }
        except Exception as e:
            print(f"Error analyzing {column}: {e}")
            continue
    
    # Perform cluster analysis
    print("Performing cluster analysis...")
    try:
        save_path = os.path.join(output_dir, 'cluster_analysis.png') if save_plots and visualize else None
        clusters = cluster_analysis(df, save_path=save_path)
        df['cluster'] = clusters
    except Exception as e:
        print(f"Error in cluster analysis: {e}")
    
    print("Analysis complete!")
    return df, results

def interpret_results(df: pd.DataFrame, results: Dict[str, Any]) -> None:
    """
    Interpret and display analysis results in a readable format.
    
    Args:
        df: Analyzed DataFrame with anomalies and clusters
        results: Dictionary containing analysis results for each metric
    """
    print("\n" + "="*80)
    print("ANALYSIS RESULTS INTERPRETATION")
    print("="*80)
    
    # Check if anomaly detection was performed
    has_anomalies = 'anomaly' in df.columns
    
    for column in ['CPU', 'GPU', 'Memory_Usage', 'HBM_Usage', 'Data_In', 'Data_Out']:
        if column not in results:
            continue
            
        print(f"\n{'─'*80}")
        print(f"Analysis for {column}")
        print(f"{'─'*80}")
        
        # Identify periods of abnormal activity
        if has_anomalies:
            abnormal_df = df[df['anomaly'] == -1]
            if len(abnormal_df) > 0:
                abnormal_dates = abnormal_df.reset_index()['Timestamp'].dt.date.unique()
                print(f"  • Anomalies detected: {len(abnormal_df)} ({len(abnormal_df)/len(df)*100:.1f}% of data)")
                if len(abnormal_dates) <= 5:
                    print(f"  • Dates with anomalies: {', '.join(map(str, abnormal_dates))}")
                else:
                    print(f"  • Dates with anomalies: {abnormal_dates[0]} to {abnormal_dates[-1]} ({len(abnormal_dates)} days)")
            else:
                print(f"  • No anomalies detected")
        
        # Analyze trend
        trend = results[column]['trend']
        trend_clean = trend.dropna()
        if len(trend_clean) > 0:
            trend_change = trend_clean.iloc[-1] - trend_clean.iloc[0]
            trend_direction = 'Increasing' if trend_change > 0 else 'Decreasing' if trend_change < 0 else 'Stable'
            print(f"  • Overall trend: {trend_direction} (change: {trend_change:+.2f})")
        
        # Analyze seasonality
        seasonal = results[column]['seasonal']
        seasonal_clean = seasonal.dropna()
        if len(seasonal_clean) > 0:
            max_seasonal_impact = seasonal_clean.abs().max()
            print(f"  • Maximum seasonal variation: ±{max_seasonal_impact:.2f}")
        
        # Analyze forecast
        forecast = results[column]['forecast']
        if len(forecast) > 0:
            current_avg = df[column].mean()
            future_avg = forecast['yhat'].tail(24).mean()
            future_change = future_avg - current_avg
            print(f"  • 24-hour forecast: {future_avg:.2f} (change: {future_change:+.2f} from current avg)")
    
    # Analyze clusters
    if 'cluster' in df.columns:
        print(f"\n{'─'*80}")
        print("Cluster Analysis")
        print(f"{'─'*80}")
        
        cluster_counts = df['cluster'].value_counts().sort_index()
        for cluster in cluster_counts.index:
            count = cluster_counts[cluster]
            print(f"\nCluster {cluster}: {count} entries ({count/len(df)*100:.1f}% of data)")
            cluster_data = df[df['cluster'] == cluster]
            
            # Show average values for each metric
            metrics_available = [col for col in ['CPU', 'GPU', 'Memory_Usage', 'HBM_Usage', 'Data_In', 'Data_Out'] 
                                if col in df.columns]
            for column in metrics_available:
                avg_val = cluster_data[column].mean()
                overall_avg = df[column].mean()
                diff = avg_val - overall_avg
                print(f"  • {column}: {avg_val:.2f} (overall avg: {overall_avg:.2f}, diff: {diff:+.2f})")

def summarize_analysis(df: pd.DataFrame, results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a concise summary of the analysis.
    
    Args:
        df: Analyzed DataFrame
        results: Dictionary containing analysis results
        
    Returns:
        Dictionary containing summary statistics
    """
    print("\n" + "="*80)
    print("EXECUTIVE SUMMARY")
    print("="*80)
    
    summary = {
        'total_records': len(df),
        'time_span': (df.index.max() - df.index.min()).total_seconds() / 3600,  # hours
        'anomalies': {},
        'trends': {},
        'clusters': {}
    }
    
    # Summary of anomalies
    if 'anomaly' in df.columns:
        total_anomalies = (df['anomaly'] == -1).sum()
        anomaly_rate = (total_anomalies / len(df) * 100)
        summary['anomalies'] = {
            'count': int(total_anomalies),
            'rate': float(anomaly_rate)
        }
        print(f"\nData Overview:")
        print(f"  • Total records: {len(df):,}")
        print(f"  • Time span: {summary['time_span']:.1f} hours")
        print(f"  • Anomalies detected: {total_anomalies} ({anomaly_rate:.2f}% of data)")

    # Summary of trends
    print(f"\nTrend Summary:")
    for column in ['CPU', 'GPU', 'Memory_Usage', 'HBM_Usage', 'Data_In', 'Data_Out']:
        if column in results and column in df.columns:
            trend = results[column]['trend']
            trend_clean = trend.dropna()
            if len(trend_clean) > 0:
                trend_change = trend_clean.iloc[-1] - trend_clean.iloc[0]
                overall_trend = 'Increasing' if trend_change > 0 else 'Decreasing' if trend_change < 0 else 'Stable'
                summary['trends'][column] = {
                    'direction': overall_trend,
                    'change': float(trend_change)
                }
                
                current_avg = df[column].mean()
                print(f"  • {column}: {overall_trend} (avg: {current_avg:.2f}, change: {trend_change:+.2f})")

    # Summary of clustering
    if 'cluster' in df.columns:
        print(f"\nCluster Distribution:")
        cluster_counts = df['cluster'].value_counts().sort_index()
        for cluster in cluster_counts.index:
            count = cluster_counts[cluster]
            percentage = count / len(df) * 100
            summary['clusters'][f'cluster_{cluster}'] = {
                'count': int(count),
                'percentage': float(percentage)
            }
            print(f"  • Cluster {cluster}: {count} records ({percentage:.1f}%)")
    
    print("\n" + "="*80)
    return summary


if __name__ == "__main__":
    import sys
    
    # Default file path
    csv_file = 'data/raw/live_log_dataset.csv'
    
    # Allow custom file path from command line
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    
    try:
        print(f"Starting analysis of {csv_file}")
        df, results = analyze_logs(csv_file, visualize=True, save_plots=False)
        interpret_results(df, results)
        summary = summarize_analysis(df, results)
        
        # Optionally save summary to JSON
        # import json
        # with open('analysis_summary.json', 'w') as f:
        #     json.dump(summary, f, indent=2)
        
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found.")
        print("Usage: python ML.py [path_to_csv_file]")
        sys.exit(1)
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
