from flask import Blueprint, render_template, jsonify, request, redirect, url_for
from backend.app.models.system_metrics import get_system_metrics
import datetime
import random
import pandas as pd
import os
import logging

# Configure logging
logger = logging.getLogger(__name__)

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@main.route('/api/metrics')
def get_metrics():
    # Get real system metrics
    metrics = get_system_metrics()
    
    # Create timestamps for the last 10 minutes
    now = datetime.datetime.now()
    timestamps = [(now - datetime.timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S") for i in range(10, -1, -1)]
    
    # Generate some simulated GPU data since psutil doesn't provide GPU metrics
    gpu_data = [random.randint(20, 80) for _ in range(11)]
    
    # Get real CPU and memory data for the current moment
    cpu_current = int(metrics['cpu']['percent'])
    memory_current = int(metrics['memory']['percent'])
    
    # Generate some historical data (simulated)
    cpu_data = [random.randint(max(0, cpu_current-20), min(100, cpu_current+20)) for _ in range(10)]
    cpu_data.append(cpu_current)  # Add current value
    
    memory_data = [random.randint(max(0, memory_current-15), min(100, memory_current+15)) for _ in range(10)]
    memory_data.append(memory_current)  # Add current value
    
    # Network data
    network_in = [metrics['network']['bytes_recv'] // 1024 - random.randint(100, 500) * i for i in range(10, -1, -1)]
    network_out = [metrics['network']['bytes_sent'] // 1024 - random.randint(50, 300) * i for i in range(10, -1, -1)]
    
    # Generate anomaly scores (mostly 0, occasionally 1)
    anomaly_scores = [0] * 11
    if random.random() < 0.2:  # 20% chance of an anomaly
        anomaly_index = random.randint(0, 10)
        anomaly_scores[anomaly_index] = 1
    
    data = {
        "timestamps": timestamps,
        "cpu": cpu_data,
        "gpu": gpu_data,
        "memory": memory_data,
        "data_in": network_in,
        "data_out": network_out,
        "anomaly_scores": anomaly_scores
    }
    
    # Save the current metrics to a CSV file for ML analysis
    save_metrics_to_csv(metrics)
    
    return jsonify(data)

def save_metrics_to_csv(metrics):
    """
    Save system metrics to a CSV file for ML analysis.
    
    Implements throttling to avoid excessive writes (max once per minute).
    Uses efficient file operations to minimize I/O overhead.
    
    Args:
        metrics: Dictionary containing system metrics
    """
    try:
        # Create a data directory if it doesn't exist
        os.makedirs('data/raw', exist_ok=True)
        
        # Define the file path
        file_path = 'data/raw/live_log_dataset.csv'
        timestamp = datetime.datetime.now()
        
        # Check if we should write (throttle to once per minute)
        file_exists = os.path.isfile(file_path)
        if file_exists:
            # Get file modification time instead of reading entire file
            file_mtime = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
            if (timestamp - file_mtime).total_seconds() < 60:
                return
        
        # Create a DataFrame with the current metrics
        data = {
            'Timestamp': timestamp,
            'CPU': metrics['cpu']['percent'],
            'Memory_Usage': metrics['memory']['percent'],
            'GPU': random.randint(20, 80),  # Simulated GPU data
            'HBM_Usage': random.randint(30, 90),  # Simulated HBM memory usage
            'Data_In': metrics['network']['bytes_recv'] // 1024,  # KB
            'Data_Out': metrics['network']['bytes_sent'] // 1024,  # KB
        }
        
        # Create a DataFrame
        df = pd.DataFrame([data])
        
        # Write to CSV with proper error handling
        df.to_csv(file_path, mode='a', header=not file_exists, index=False)
        
    except (OSError, IOError) as e:
        # Log the error but don't fail the entire request
        logger.warning(f"Failed to save metrics to CSV: {e}")

@main.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            # Save the file to the data directory
            file_path = 'data/raw/' + file.filename
            file.save(file_path)
            return redirect(url_for('main.dashboard'))
    return render_template('upload.html')