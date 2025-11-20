"""
Pytest configuration file for the Real-Time System Monitoring & ML Model Analysis project.

This file contains fixtures and configuration that are shared across all tests.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
