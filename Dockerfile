# Minimal LLM Training Infrastructure
# Dockerfile for GPU training on Brev

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Set working directory
WORKDIR /workspace/project/inazuma

# Copy project files (without data)
COPY pyproject.toml uv.lock ./
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY main.py ./

# Create .venv and install dependencies
RUN uv venv --python 3.11
RUN uv sync

# Create data directories
RUN mkdir -p data/raw data/processed data/cache checkpoints logs

# Set Python path
ENV PYTHONPATH="/workspace/project/inazuma:${PYTHONPATH}"

# Default command
CMD ["bash"]