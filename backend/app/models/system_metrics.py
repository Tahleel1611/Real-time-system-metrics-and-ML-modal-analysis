"""
System metrics collection module for real-time monitoring.

This module provides efficient system metrics collection with caching
to minimize redundant psutil calls and improve performance.
"""
import psutil
from datetime import datetime
from typing import Dict, Any


def get_system_metrics() -> Dict[str, Any]:
    """
    Collect current system metrics including CPU, memory, disk, and network usage.
    
    Returns:
        Dict containing system metrics with the following structure:
        {
            'cpu': {'percent': float, 'count': int},
            'memory': {'total': int, 'available': int, 'percent': float},
            'disk': {'total': int, 'used': int, 'percent': float},
            'network': {'bytes_sent': int, 'bytes_recv': int},
            'timestamp': str (ISO format)
        }
    """
    # Cache psutil calls to avoid redundant system queries
    memory_info = psutil.virtual_memory()
    disk_info = psutil.disk_usage('/')
    network_info = psutil.net_io_counters()
    
    metrics = {
        'cpu': {
            'percent': psutil.cpu_percent(interval=1),
            'count': psutil.cpu_count(logical=True)
        },
        'memory': {
            'total': memory_info.total,
            'available': memory_info.available,
            'percent': memory_info.percent
        },
        'disk': {
            'total': disk_info.total,
            'used': disk_info.used,
            'percent': disk_info.percent
        },
        'network': {
            'bytes_sent': network_info.bytes_sent,
            'bytes_recv': network_info.bytes_recv
        },
        'timestamp': datetime.now().isoformat()
    }
    return metrics

def get_process_info() -> list:
    """
    Retrieve information about all running processes.
    
    Returns:
        List of dictionaries containing process information:
        [
            {
                'pid': int,
                'name': str,
                'cpu_percent': float,
                'memory': int (MB),
                'disk_read': int (bytes),
                'disk_write': int (bytes)
            },
            ...
        ]
    """
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info', 'io_counters']):
        try:
            proc_info = proc.info
            io_counters = proc_info.get('io_counters')
            
            processes.append({
                'pid': proc_info['pid'],
                'name': proc_info['name'],
                'cpu_percent': proc_info['cpu_percent'] or 0.0,
                'memory': proc_info['memory_info'].rss // (1024 * 1024),  # Memory in MB
                'disk_read': io_counters.read_bytes if io_counters else 0,
                'disk_write': io_counters.write_bytes if io_counters else 0,
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
            continue
    return processes