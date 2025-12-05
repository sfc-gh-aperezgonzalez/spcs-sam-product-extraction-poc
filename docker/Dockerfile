# SAM 2 Detection Service - HuggingFace Implementation
#
# Build from project root:
#   docker build -f docker/Dockerfile -t sam2-detector:latest .
#
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

WORKDIR /app

# Install PyTorch 2.5+ (required for transformers SAM2 support)
RUN pip3 install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu118

# Install HuggingFace transformers with SAM2 support
RUN pip3 install --no-cache-dir \
    "transformers>=4.46.0" \
    accelerate \
    safetensors \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    Pillow \
    numpy

# Copy pre-downloaded HuggingFace model (for offline use)
COPY docker/hf_models/sam2.1-hiera-small /app/hf_models/sam2.1-hiera-small

# Copy application code
COPY app/ /app/

ENV PYTHONUNBUFFERED=1
ENV MODEL_ID=/app/hf_models/sam2.1-hiera-small
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_OFFLINE=1
ENV PORT=8080

EXPOSE 8080

CMD ["python3", "main.py"]
