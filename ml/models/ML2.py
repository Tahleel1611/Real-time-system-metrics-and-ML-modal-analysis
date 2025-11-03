"""
Advanced ML analysis module with correlation and classification features.

This module extends ML.py with additional analysis capabilities:
- Optimal cluster number detection using silhouette scores
- Correlation analysis between metrics
- Anomaly classification with Random Forest
- Feature importance analysis

Refactored to use common functions from ML.py to reduce code duplication.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, classification_report
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple, Any
import warnings

# Import base functions from ML.py to avoid duplication
from ml.models.ML import (
    load_and_prepare_data,
    detect_anomalies,
    analyze_time_series,
    forecast_with_prophet
)

warnings.filterwarnings('ignore', category=FutureWarning)

def numeric_cluster_analysis(df: pd.DataFrame, max_clusters: int = 10) -> Tuple[np.ndarray, pd.DataFrame, int]:
    """
    Perform clustering analysis with automatic optimal cluster number detection.
    
    Uses silhouette score to determine the optimal number of clusters.
    
    Args:
        df: DataFrame containing metrics data
        max_clusters: Maximum number of clusters to test (default: 10)
        
    Returns:
        Tuple of (cluster_labels, cluster_statistics, optimal_n_clusters)
    """
    # Prepare data for clustering
    df_clean = df.drop('anomaly', axis=1) if 'anomaly' in df.columns else df.copy()
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_clean)
    
    # Determine optimal number of clusters using silhouette score
    print("Determining optimal number of clusters...")
    silhouette_scores = []
    cluster_range = range(2, min(max_clusters + 1, len(df_clean)))
    
    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_data)
        score = silhouette_score(scaled_data, cluster_labels)
        silhouette_scores.append(score)
        print(f"  n_clusters={n_clusters}: silhouette_score={score:.4f}")
    
    # Find optimal number of clusters
    optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
    print(f"Optimal number of clusters: {optimal_clusters}")
    
    # Perform clustering with optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_data)
    
    # Create a copy to add cluster column
    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = clusters
    
    # Analyze cluster characteristics
    cluster_stats = df_with_clusters.groupby('cluster').agg(['mean', 'std'])
    
    return clusters, cluster_stats, optimal_clusters

def correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute correlation matrix for metrics.
    
    Args:
        df: DataFrame containing metrics data
        
    Returns:
        Correlation matrix as DataFrame
    """
    # Exclude non-numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    return corr_matrix


def anomaly_classification(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    """
    Train a Random Forest classifier to predict anomalies.
    
    Args:
        df: DataFrame with 'anomaly' and 'cluster' columns
        test_size: Proportion of data to use for testing (default: 0.2)
        
    Returns:
        Tuple of (trained_classifier, classification_report_dict)
        
    Raises:
        ValueError: If required columns are missing
    """
    if 'anomaly' not in df.columns:
        raise ValueError("DataFrame must contain 'anomaly' column")
    
    # Prepare features and labels
    exclude_cols = ['anomaly']
    if 'cluster' in df.columns:
        exclude_cols.append('cluster')
    
    features = df.drop(exclude_cols, axis=1)
    labels = df['anomaly']
    
    # Check if we have enough samples
    if len(df) < 10:
        raise ValueError("Insufficient data for classification. Need at least 10 samples.")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    # Train Random Forest classifier
    print(f"Training anomaly classifier on {len(X_train)} samples...")
    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        max_depth=10,
        min_samples_split=5
    )
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    classification_rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    print(f"Classification accuracy: {classification_rep['accuracy']:.4f}")
    
    return clf, classification_rep

def analyze_logs(file_path: str) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame, 
                                         RandomForestClassifier, pd.DataFrame, Dict[str, Any], int]:
    """
    Perform comprehensive advanced ML analysis on system log data.
    
    This function extends the basic ML.py analysis with:
    - Optimal cluster number detection
    - Correlation analysis
    - Anomaly classification
    
    Args:
        file_path: Path to CSV file containing log data
        
    Returns:
        Tuple of (dataframe, results, correlation_matrix, classifier, 
                 cluster_stats, classification_report, optimal_clusters)
    """
    print(f"Loading data from {file_path}...")
    df = load_and_prepare_data(file_path)
    
    print("Detecting anomalies...")
    df = detect_anomalies(df)
    
    results = {}
    columns_to_analyze = ['CPU', 'GPU', 'Memory_Usage', 'HBM_Usage', 'Data_In', 'Data_Out']
    
    print("Performing time series analysis and forecasting...")
    for column in columns_to_analyze:
        if column not in df.columns:
            print(f"Warning: Column '{column}' not found. Skipping...")
            continue
        
        try:
            trend, seasonal, residual = analyze_time_series(df, column)
            forecast = forecast_with_prophet(df, column)
            results[column] = {
                'trend': trend,
                'seasonal': seasonal,
                'residual': residual,
                'forecast': forecast
            }
        except Exception as e:
            print(f"Error analyzing {column}: {e}")
            continue
    
    print("\nPerforming advanced cluster analysis...")
    clusters, cluster_stats, optimal_clusters = numeric_cluster_analysis(df)
    df['cluster'] = clusters
    
    print("\nPerforming correlation analysis...")
    corr_matrix = correlation_analysis(df)
    
    print("\nTraining anomaly classifier...")
    try:
        anomaly_classifier, classification_rep = anomaly_classification(df)
    except Exception as e:
        print(f"Error in anomaly classification: {e}")
        anomaly_classifier = None
        classification_rep = {}
    
    print("\nAdvanced analysis complete!")
    return df, results, corr_matrix, anomaly_classifier, cluster_stats, classification_rep, optimal_clusters

def interpret_results(df: pd.DataFrame, results: Dict[str, Any], corr_matrix: pd.DataFrame,
                     anomaly_classifier: RandomForestClassifier, cluster_stats: pd.DataFrame,
                     classification_rep: Dict[str, Any], optimal_clusters: int) -> None:
    """
    Interpret and display advanced analysis results.
    
    Args:
        df: Analyzed DataFrame
        results: Dictionary containing analysis results for each metric
        corr_matrix: Correlation matrix
        anomaly_classifier: Trained Random Forest classifier
        cluster_stats: Cluster statistics DataFrame
        classification_rep: Classification report dictionary
        optimal_clusters: Optimal number of clusters found
    """
    print("\n" + "="*80)
    print("ADVANCED ANALYSIS RESULTS")
    print("="*80)
    
    # Time series analysis
    print("\nTime Series Analysis:")
    print("─"*80)
    for column in ['CPU', 'GPU', 'Memory_Usage', 'HBM_Usage', 'Data_In', 'Data_Out']:
        if column not in results:
            continue
            
        print(f"\n{column}:")
        trend = results[column]['trend']
        trend_clean = trend.dropna()
        if len(trend_clean) > 0:
            trend_change = trend_clean.iloc[-1] - trend_clean.iloc[0]
            print(f"  • Trend: {'Increasing' if trend_change > 0 else 'Decreasing'} ({trend_change:+.2f})")
        
        seasonal = results[column]['seasonal']
        seasonal_clean = seasonal.dropna()
        if len(seasonal_clean) > 0:
            max_seasonal_impact = seasonal_clean.abs().max()
            print(f"  • Max seasonal variation: ±{max_seasonal_impact:.2f}")
        
        forecast = results[column]['forecast']
        if len(forecast) > 0:
            future_trend = forecast['yhat'].iloc[-1] - forecast['yhat'].iloc[0]
            print(f"  • Forecast trend: {'Increasing' if future_trend > 0 else 'Decreasing'}")
    
    # Correlation analysis
    print("\n" + "─"*80)
    print("Correlation Analysis:")
    print("─"*80)
    strong_correlations = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            correlation = corr_matrix.iloc[i, j]
            if abs(correlation) > 0.5:
                strong_correlations.append({
                    'metric1': corr_matrix.columns[i],
                    'metric2': corr_matrix.columns[j],
                    'correlation': correlation
                })
    
    if strong_correlations:
        for corr in sorted(strong_correlations, key=lambda x: abs(x['correlation']), reverse=True):
            print(f"  • {corr['metric1']} ↔ {corr['metric2']}: {corr['correlation']:+.3f}")
    else:
        print("  • No strong correlations (|r| > 0.5) found")
    
    # Cluster analysis
    print("\n" + "─"*80)
    print(f"Cluster Analysis (Optimal clusters: {optimal_clusters}):")
    print("─"*80)
    print(cluster_stats.to_string())
    
    # Anomaly classification
    if classification_rep and anomaly_classifier:
        print("\n" + "─"*80)
        print("Anomaly Classification Performance:")
        print("─"*80)
        print(f"  • Accuracy: {classification_rep.get('accuracy', 0):.4f}")
        if 'weighted avg' in classification_rep:
            print(f"  • Precision: {classification_rep['weighted avg']['precision']:.4f}")
            print(f"  • Recall: {classification_rep['weighted avg']['recall']:.4f}")
            print(f"  • F1-score: {classification_rep['weighted avg']['f1-score']:.4f}")
        
        # Feature importance
        if hasattr(anomaly_classifier, 'feature_importances_'):
            feature_cols = [col for col in df.columns if col not in ['anomaly', 'cluster']]
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': anomaly_classifier.feature_importances_
            })
            feature_importance = feature_importance.sort_values('importance', ascending=False)
            print("\n  Top 3 features for anomaly detection:")
            for idx, row in feature_importance.head(3).iterrows():
                print(f"    {idx+1}. {row['feature']}: {row['importance']:.4f}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    import sys
    
    # Default file path
    csv_file = 'data/raw/live_log_dataset.csv'
    
    # Allow custom file path from command line
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    
    try:
        print(f"Starting advanced analysis of {csv_file}")
        df, results, corr_matrix, anomaly_classifier, cluster_stats, classification_rep, optimal_clusters = analyze_logs(csv_file)
        interpret_results(df, results, corr_matrix, anomaly_classifier, cluster_stats, classification_rep, optimal_clusters)
        
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found.")
        print("Usage: python ML2.py [path_to_csv_file]")
        sys.exit(1)
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
