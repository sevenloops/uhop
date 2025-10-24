# Dockerfile for UHOP CLI/Development
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Install Python package
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Default command runs CLI help
CMD ["uhop", "--help"]
