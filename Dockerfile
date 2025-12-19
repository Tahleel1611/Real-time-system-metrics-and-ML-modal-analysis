# Multi-stage Dockerfile for HPC System Monitoring Platform
# Optimized for production deployment with GPU support

# Stage 1: Base image with Python and system dependencies
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Dependencies installation
FROM base as dependencies

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 3: Final production image
FROM base as production

WORKDIR /app

# Copy installed dependencies from previous stage
COPY --from=dependencies /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# Copy application code
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY ml/ ./ml/
COPY data/ ./data/

# Create necessary directories
RUN mkdir -p /app/logs /app/uploads /app/reports

# Set proper permissions
RUN chmod -R 755 /app

# Expose Flask port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

# Run the application
CMD ["python", "backend/app/app.py"]
