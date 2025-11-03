"""
Process and system information collection module.

This module provides comprehensive system monitoring functionality including
CPU, memory, disk, network, GPU, and process information.
"""
import psutil
import platform
import subprocess
from typing import Dict, List, Any, Tuple, Optional, Union

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


def get_cpu_performance() -> Tuple[Union[float, str], Optional[float], Optional[float], Optional[float]]:
    """
    Get CPU performance metrics including utilization and frequency.
    
    Returns:
        Tuple of (cpu_percent, current_freq, min_freq, max_freq)
        Returns error message string for cpu_percent if operation fails.
    """
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_freq = psutil.cpu_freq()
        if cpu_freq:
            return cpu_percent, cpu_freq.current, cpu_freq.min, cpu_freq.max
        return cpu_percent, None, None, None
    except Exception as e:
        return f"Error retrieving CPU performance: {e}", None, None, None

def get_memory_info() -> Tuple[Union[float, str], Optional[float], Optional[float]]:
    """
    Get memory usage information.
    
    Returns:
        Tuple of (percent_used, total_gb, available_gb)
    """
    try:
        ram = psutil.virtual_memory()
        return ram.percent, ram.total / (1024 ** 3), ram.available / (1024 ** 3)
    except Exception as e:
        return f"Error retrieving memory info: {e}", None, None


def get_disk_info() -> Tuple[Union[float, str], Optional[float], Optional[float]]:
    """
    Get disk usage information for root partition.
    
    Returns:
        Tuple of (percent_used, total_gb, free_gb)
    """
    try:
        disk = psutil.disk_usage('/')
        return disk.percent, disk.total / (1024 ** 3), disk.free / (1024 ** 3)
    except Exception as e:
        return f"Error retrieving disk info: {e}", None, None


# Cache for network stats to calculate speed without blocking
_last_network_stats = None
_last_network_time = None


def get_network_speed() -> Tuple[Union[float, str], Union[float, str]]:
    """
    Get network upload and download speed in KB/s.
    
    Uses cached values from previous call to avoid blocking with time.sleep().
    First call returns (0, 0) as baseline.
    
    Returns:
        Tuple of (download_speed_kbps, upload_speed_kbps)
    """
    global _last_network_stats, _last_network_time
    
    try:
        import time
        current_time = time.time()
        current_stats = psutil.net_io_counters()
        
        if _last_network_stats is None or _last_network_time is None:
            # First call - establish baseline
            _last_network_stats = current_stats
            _last_network_time = current_time
            return 0.0, 0.0
        
        # Calculate time difference
        time_delta = current_time - _last_network_time
        if time_delta < 0.1:  # Avoid division by very small numbers
            return 0.0, 0.0
        
        # Calculate speeds
        download_speed = (current_stats.bytes_recv - _last_network_stats.bytes_recv) / 1024 / time_delta
        upload_speed = (current_stats.bytes_sent - _last_network_stats.bytes_sent) / 1024 / time_delta
        
        # Update cache
        _last_network_stats = current_stats
        _last_network_time = current_time
        
        return download_speed, upload_speed
    except Exception as e:
        error_msg = f"Error retrieving network speed: {e}"
        return error_msg, error_msg

def get_gpu_info() -> Union[List[Dict[str, Any]], str]:
    """
    Get GPU information for all available GPUs.
    
    Returns:
        List of dictionaries containing GPU metrics, or error message string.
        Each dictionary contains: id, name, load, memory_used, memory_total, temperature
    """
    if not GPU_AVAILABLE:
        return "GPUtil not installed. Install with: pip install gputil"
    
    try:
        gpus = GPUtil.getGPUs()
        if not gpus:
            return "No GPU detected."
        
        gpu_data = []
        for gpu in gpus:
            gpu_data.append({
                'id': gpu.id,
                'name': gpu.name,
                'load': gpu.load * 100,
                'memory_used': gpu.memoryUsed,
                'memory_total': gpu.memoryTotal,
                'temperature': gpu.temperature
            })
        return gpu_data
    except Exception as e:
        return f"Error retrieving GPU info: {e}"

def get_system_logs() -> List[str]:
    """
    Get recent system logs (platform-specific).
    
    Returns:
        List of log entries as strings, or error messages if retrieval fails.
    """
    try:
        logs = []
        system = platform.system()
        
        if system == "Linux":
            # Use safer subprocess call without shell
            result = subprocess.run(
                ["dmesg"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                # Process output to get last 10 lines using Python
                logs = result.stdout.splitlines()[-10:]
            else:
                logs = [result.stderr]
        elif system == "Windows":
            # Use safer subprocess call without shell=True
            result = subprocess.run(
                ["powershell", "-Command", "Get-EventLog -LogName System -Newest 10"],
                capture_output=True,
                text=True,
                timeout=5
            )
            logs = result.stdout.splitlines() if result.returncode == 0 else [result.stderr]
        else:
            logs = [f"System logs not supported on {system}"]
        
        return logs
    except subprocess.TimeoutExpired:
        return ["Error: Log retrieval timed out"]
    except Exception as e:
        return [f"Error retrieving system logs: {e}"]

def get_process_info() -> List[Dict[str, Any]]:
    """
    Get information about all running processes.
    
    Returns:
        List of dictionaries containing process information (pid, name, cpu_percent, memory_info).
    """
    try:
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info']):
            try:
                processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return processes
    except Exception as e:
        return [{"error": f"Error retrieving process info: {e}"}]

def get_temperature_info() -> Union[Dict, str]:
    """
    Get system temperature sensor readings.
    
    Returns:
        Dictionary of temperature sensors or error message string.
    """
    try:
        temp_info = psutil.sensors_temperatures()
        if not temp_info:
            return "No temperature sensors found."
        # Return coretemp if available, otherwise return all available sensors
        return temp_info.get('coretemp', temp_info)
    except AttributeError:
        return "Temperature sensors not supported on this platform."
    except Exception as e:
        return f"Error retrieving temperature info: {e}"

def get_services_info() -> List[str]:
    """
    Get list of running services (platform-specific).
    
    Returns:
        List of running service names/descriptions.
    """
    try:
        services = []
        system = platform.system()
        
        if system == "Linux":
            result = subprocess.run(
                ["systemctl", "list-units", "--type=service", "--state=running", "--no-pager"],
                capture_output=True,
                text=True,
                timeout=5
            )
            services = result.stdout.splitlines() if result.returncode == 0 else [result.stderr]
        elif system == "Windows":
            result = subprocess.run(
                ["powershell", "-Command", "Get-Service | Where-Object {$_.Status -eq 'Running'} | Select-Object -First 20"],
                capture_output=True,
                text=True,
                timeout=5
            )
            services = result.stdout.splitlines() if result.returncode == 0 else [result.stderr]
        else:
            services = [f"Service listing not supported on {system}"]
        
        return services
    except subprocess.TimeoutExpired:
        return ["Error: Service listing timed out"]
    except Exception as e:
        return [f"Error retrieving services info: {e}"]

def get_applications_info() -> List[str]:
    """
    Get list of all running application names.
    
    Returns:
        List of application names.
    """
    try:
        apps = []
        for proc in psutil.process_iter(['name']):
            try:
                if proc.info['name']:
                    apps.append(proc.info['name'])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        # Return unique application names
        return list(set(apps))
    except Exception as e:
        return [f"Error retrieving applications info: {e}"]


def get_battery_info() -> Tuple[Union[float, str, None], Optional[bool]]:
    """
    Get battery status information.
    
    Returns:
        Tuple of (battery_percent, is_plugged) or (None, None) if no battery.
    """
    try:
        battery = psutil.sensors_battery()
        if battery:
            return battery.percent, battery.power_plugged
        else:
            return None, None
    except AttributeError:
        # sensors_battery not available on this platform
        return None, None
    except Exception as e:
        return f"Error retrieving battery info: {e}", None


def get_system_support_info() -> Union[Dict[str, Any], str]:
    """
    Get system platform and architecture information.
    
    Returns:
        Dictionary containing system information or error message string.
    """
    try:
        support_info = {
            'platform': platform.platform(),
            'architecture': platform.architecture(),
            'system': platform.system(),
            'node': platform.node(),
            'release': platform.release(),
            'version': platform.version()
        }
        return support_info
    except Exception as e:
        return f"Error retrieving system support info: {e}"

def print_system_info(refresh_interval: int = 10) -> None:
    """
    Continuously print system information to console.
    
    Args:
        refresh_interval: Seconds between updates (default: 10)
    """
    import time
    
    print("Fetching system information... Press Ctrl+C to stop.")
    try:
        while True:
            # Collect all system information
            cpu_performance = get_cpu_performance()
            memory_info = get_memory_info()
            disk_info = get_disk_info()
            network_speed = get_network_speed()
            gpu_info = get_gpu_info()
            system_logs = get_system_logs()
            process_info = get_process_info()
            temperature_info = get_temperature_info()
            services_info = get_services_info()
            applications_info = get_applications_info()
            battery_info = get_battery_info()
            support_info = get_system_support_info()
            
            # Display CPU information
            if isinstance(cpu_performance[0], (int, float)):
                print(f"\nCPU Utilization: {cpu_performance[0]:.1f}%")
                if cpu_performance[1]:
                    print(f"CPU Frequency: {cpu_performance[1]:.0f} MHz (Min: {cpu_performance[2]:.0f} MHz, Max: {cpu_performance[3]:.0f} MHz)")
            else:
                print(f"\n{cpu_performance[0]}")

            # Display memory information
            if isinstance(memory_info[0], (int, float)):
                print(f"Memory Utilization: {memory_info[0]:.1f}%, Total: {memory_info[1]:.2f} GB, Available: {memory_info[2]:.2f} GB")
            else:
                print(memory_info[0])

            # Display disk information
            if isinstance(disk_info[0], (int, float)):
                print(f"Disk Utilization: {disk_info[0]:.1f}%, Total: {disk_info[1]:.2f} GB, Free: {disk_info[2]:.2f} GB")
            else:
                print(disk_info[0])

            # Display network speed
            if isinstance(network_speed[0], (int, float)) and isinstance(network_speed[1], (int, float)):
                print(f"Network Speed: Download: {network_speed[0]:.2f} KB/s, Upload: {network_speed[1]:.2f} KB/s")
            else:
                print(f"Network Speed: {network_speed[0]}")

            # Display GPU information
            if isinstance(gpu_info, str):
                print(f"\n{gpu_info}")
            else:
                for gpu in gpu_info:
                    print(f"\nGPU {gpu['id']} - {gpu['name']}:")
                    print(f"  Utilization: {gpu['load']:.1f}%")
                    print(f"  Memory Used: {gpu['memory_used']:.0f} MB / {gpu['memory_total']:.0f} MB")
                    print(f"  Temperature: {gpu['temperature']:.1f} °C")

            # Display system logs (first 5 only)
            print("\nRecent System Logs:")
            for log in system_logs[:5]:
                print(f"  {log}")

            # Display top processes by CPU (first 5 only)
            print("\nTop Processes by CPU:")
            if process_info and not isinstance(process_info[0], dict) or 'error' not in process_info[0]:
                sorted_procs = sorted(
                    [p for p in process_info if 'cpu_percent' in p and p['cpu_percent']],
                    key=lambda x: x.get('cpu_percent', 0),
                    reverse=True
                )[:5]
                for proc in sorted_procs:
                    if 'memory_info' in proc and proc['memory_info']:
                        mem_mb = proc['memory_info'].rss / (1024 ** 2)
                        print(f"  PID: {proc['pid']}, Name: {proc['name']}, CPU: {proc['cpu_percent']:.1f}%, Memory: {mem_mb:.2f} MB")

            # Display temperature information
            print("\nTemperature Info:")
            if isinstance(temperature_info, str):
                print(f"  {temperature_info}")
            elif isinstance(temperature_info, dict):
                for sensor_name, readings in list(temperature_info.items())[:3]:  # Limit to 3 sensors
                    if isinstance(readings, list):
                        for reading in readings[:2]:  # Limit to 2 readings per sensor
                            label = reading.label if hasattr(reading, 'label') else sensor_name
                            print(f"  {label}: {reading.current:.1f} °C")

            # Display battery info
            if battery_info[0] is not None and not isinstance(battery_info[0], str):
                print(f"\nBattery: {battery_info[0]:.1f}%, Plugged In: {'Yes' if battery_info[1] else 'No'}")
            elif isinstance(battery_info[0], str):
                print(f"\n{battery_info[0]}")

            # Display system support information
            if isinstance(support_info, dict):
                print("\nSystem Information:")
                print(f"  Platform: {support_info.get('platform', 'Unknown')}")
                print(f"  System: {support_info.get('system', 'Unknown')}")
                print(f"  Release: {support_info.get('release', 'Unknown')}")

            print(f"\n{'='*60}")
            print(f"Next update in {refresh_interval} seconds...")
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\n\nStopped monitoring.")


if __name__ == "__main__":
    print_system_info()
