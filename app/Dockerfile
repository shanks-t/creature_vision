# Start with a base image that supports both amd64 and arm64
FROM --platform=$BUILDPLATFORM python:3.10-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m -s /bin/bash app
WORKDIR /app
USER app

# Install Python dependencies
COPY --chown=app:app requirements.txt .
RUN pip install --user -r requirements.txt

# Copy application code
COPY --chown=app:app . .

# Set up environment for different architectures
FROM base as amd64
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib64

FROM base as arm64

# Use multi-stage build to create the final image
FROM ${TARGETARCH}

# Install TensorFlow Metal for M1/M2 Macs (only for arm64)
RUN if [ "$(uname -m)" = "arm64" ]; then \
    pip install --user tensorflow-macos tensorflow-metal; \
    fi

# Set the entrypoint
CMD ["python", "app.py"]