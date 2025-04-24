from flask import Blueprint, render_template, jsonify, request, redirect, url_for
from backend.app.models.system_metrics import get_system_metrics
import datetime
import random
import pandas as pd
import os
import time

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
    Save system metrics to a CSV file for ML analysis
    """
    # Create a data directory if it doesn't exist
    os.makedirs('data/raw', exist_ok=True)
    
    # Create a DataFrame with the current metrics
    timestamp = datetime.datetime.now()
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
    
    # Define the file path
    file_path = 'data/raw/live_log_dataset.csv'
    
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.isfile(file_path)
    
    # Append to the CSV file
    if file_exists:
        # Read existing file to check if it's been more than 10 minutes since the last update
        existing_df = pd.read_csv(file_path)
        if len(existing_df) > 0:
            last_timestamp = pd.to_datetime(existing_df['Timestamp'].iloc[-1])
            if (timestamp - last_timestamp).total_seconds() < 60:  # Only append every minute
                return
    
    # Write to CSV
    df.to_csv(file_path, mode='a', header=not file_exists, index=False)

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