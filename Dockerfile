FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl wget ca-certificates git \
    ffmpeg libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy package definition and requirements first for better caching
COPY pyproject.toml .
COPY requirements.txt .
COPY src/ src/

# Install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy application code and assets
COPY app.py .
COPY static/ static/
COPY assets/ assets/

# Expose HF Spaces default port
EXPOSE 7860

# Run the application
CMD ["python", "app.py"]
