# ============================================================================
# Dockerfile for SAM (Segment Anything Model) Inference Service
# Base: NVIDIA CUDA 12.1 with cuDNN for GPU acceleration
# ============================================================================

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.10 and essential build tools
RUN apt-get update && \
    apt-get install -y \
        python3.10 \
        python3.10-venv \
        python3-pip \
        git \
        wget \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --no-cache-dir --upgrade pip

# Install PyTorch with CUDA 12.1 support (AMD64 compatible versions)
RUN pip install --no-cache-dir \
    torch==2.1.0+cu121 \
    torchvision==0.16.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Install Segment Anything Model (SAM)
RUN pip install --no-cache-dir git+https://github.com/facebookresearch/segment-anything.git

# Install core dependencies
COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Set working directory
WORKDIR /app

# Copy application code (from parent directory)
# Note: Build from project root with: docker build -f docker/Dockerfile .
COPY app/ /app/

# Expose FastAPI port
ENV PORT=8080
EXPOSE 8080

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:8080/health')" || exit 1

# Run FastAPI with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
