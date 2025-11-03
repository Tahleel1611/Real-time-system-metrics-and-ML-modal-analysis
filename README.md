# Real-Time System Monitoring & ML Model Analysis

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code Quality](https://img.shields.io/badge/code%20quality-optimized-brightgreen.svg)](https://github.com/Tahleel1611/Real-time-system-metrics-and-ML-modal-analysis)

> **Professional real-time system monitoring and machine learning model analysis platform for High-Performance Computing (HPC) environments**

## Overview

In Machine Learning and High-Performance Computing environments, monitoring system performance, hardware utilization, and training process efficiency is critical. This platform provides a dynamic, interactive interface for real-time data visualization, historical trend analysis, and comprehensive insights into system components including CPU, GPU, memory, HBM memory, and more.

## Key Features

### 🖥️ Real-Time System Monitoring
- **Live Metrics Visualization**: Monitor CPU, GPU, memory, storage, and network performance in real-time
- **System Logs Analysis**: Access and analyze system logs with intelligent filtering
- **Hardware Utilization Tracking**: Detailed CPU, GPU, HBM memory usage statistics
- **Process Monitoring**: Track individual process resource consumption

### 🤖 Machine Learning Analytics
- **Anomaly Detection**: Identify unusual patterns in system metrics using Isolation Forest algorithm
- **Performance Forecasting**: Predict future system behavior with Prophet time series models
- **Trend Analysis**: Understand long-term patterns with seasonal decomposition
- **Clustering Analysis**: Group similar system states for pattern recognition
- **Interactive Visualizations**: Comprehensive charts and graphs for data interpretation

### 📊 Historical Data Analysis
- **Time-Travel Capability**: Navigate through historical system data
- **Training Progress Visualization**: Monitor ML model training efficiency over time
- **Performance Correlation**: Identify relationships between different system metrics

## Screenshots

Here's a preview of the platform:







![Main Dashboard](https://github.com/user-attachments/assets/350e5691-2f87-4ca5-b3d3-4bcbb29d4d4d)
*Main Dashboard - Real-time system metrics overview*

![Performance Analytics](https://github.com/user-attachments/assets/4f124b09-ce39-4899-a6bd-1cbffefde1c8)
*Performance Analytics - Detailed CPU and GPU monitoring*

![ML Analysis](https://github.com/user-attachments/assets/2567af8a-ca1d-4638-a4a1-dd84b4b515d0)
*ML Analysis - Anomaly detection and forecasting*

![Historical View](https://github.com/user-attachments/assets/cdc6e6e3-0886-43d4-8c52-8eef7869a425)
*Historical Data View - Time series analysis*

![Cluster Analysis](https://github.com/user-attachments/assets/9f96a68f-6e06-4cb3-9aeb-932691ff4c6c)
*Cluster Analysis - System state grouping*

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Frontend      │────▶│   Flask Backend  │────▶│  ML Models      │
│   (HTML/CSS/JS) │     │   (REST API)     │     │  (Scikit-learn) │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │                           │
                               ▼                           ▼
                        ┌──────────────┐          ┌──────────────┐
                        │   psutil     │          │   Prophet    │
                        │   (Metrics)  │          │   (Forecast) │
                        └──────────────┘          └──────────────┘
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/Tahleel1611/Real-time-system-metrics-and-ML-modal-analysis.git
   cd Real-time-system-metrics-and-ML-modal-analysis
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   # On Windows
   run_project.bat
   
   # On macOS/Linux
   python backend/app/app.py
   ```

5. **Access the application**
   - Open your web browser
   - Navigate to `http://localhost:5000`

## Usage

### Basic Monitoring
1. Launch the application
2. Navigate to the dashboard to view real-time metrics
3. Monitor CPU, GPU, memory, and network utilization

### ML Analysis
1. Upload historical log data (CSV format) via the upload interface
2. Navigate to the ML analysis section
3. View anomaly detection results, forecasts, and clustering analysis

### Command-Line Tools

**Run ML analysis on log data:**
```bash
python ml/models/ML.py path/to/logfile.csv
```

**Run anomaly detection:**
```bash
python ml/models/anomaly_detection.py
```

**Monitor system in console:**
```bash
python backend/app/models/processes.py
```

## Configuration

Edit `backend/config.py` to customize:
- Database settings
- File upload limits
- ML model parameters
- Performance tuning options

## Performance Optimizations

This project includes several performance improvements:

✅ **Efficient System Queries**: Cached psutil calls to reduce redundant system queries  
✅ **Optimized CSV Operations**: File modification time checks instead of full file reads  
✅ **Parallel Processing**: Multi-threaded ML model training (n_jobs=-1)  
✅ **Non-blocking Operations**: Removed time.sleep() calls in critical paths  
✅ **Secure Subprocess Calls**: Eliminated shell=True for better security  
✅ **Type Hints & Documentation**: Comprehensive docstrings and type annotations  
✅ **Error Handling**: Robust exception handling throughout the codebase  

## Project Structure

```
Real-time-system-metrics-and-ML-modal-analysis/
├── backend/
│   ├── app/
│   │   ├── models/          # System metrics collection
│   │   ├── routes.py        # API endpoints
│   │   └── __init__.py      # Flask app factory
│   └── config.py            # Configuration management
├── ml/
│   ├── models/
│   │   ├── ML.py            # Comprehensive ML analysis
│   │   ├── anomaly_detection.py  # Anomaly detection
│   │   └── forecasting.py   # Time series forecasting
│   └── utils/               # ML utilities
├── frontend/
│   ├── templates/           # HTML templates
│   └── static/              # CSS, JS, assets
├── tests/                   # Unit tests
├── data/                    # Data storage
└── requirements.txt         # Python dependencies
```

## Dependencies

- **Flask**: Web framework
- **psutil**: System metrics collection
- **pandas**: Data manipulation
- **scikit-learn**: Machine learning algorithms
- **Prophet**: Time series forecasting
- **matplotlib**: Data visualization
- **GPUtil**: GPU monitoring (optional)

## Contributing

Contributions are welcome! Please follow these guidelines:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built for High-Performance Computing environments
- Optimized for machine learning workload monitoring
- Designed with scalability and performance in mind

## Support

For issues, questions, or contributions, please open an issue on GitHub or contact the maintainers.

---

**Made with ❤️ for the HPC and ML community**
