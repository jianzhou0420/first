FROM docker.aiml.team/products/aiml/docker-example/pytorch-2.3.1-cuda12.1-cudnn8-devel:latest

# Install dependencies and clean up in a single layer
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    curl \
    libglib2.0-0 \
    git \
    ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install Python packages

# (Optional) Set the working directory
WORKDIR /workspace
