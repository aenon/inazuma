# Minimal Language Model Training Infrastructure

A minimal yet functional infrastructure for training language models using knowledge distillation from a teacher model.

## Quick Start

```bash
# 1. Install dependencies and setup
make setup

# 2. Verify environment
make verify

# 3. Run training
make train
```

## Using Makefile

```bash
# Setup & Installation
make install          # Install dependencies (uv sync)
make setup           # Full setup (install + download + verify)
make verify          # Verify environment

# Training
make train           # Run training with CONFIG=configs/train.yaml

# Docker
make docker-build    # Build container image
make docker-verify # Verify Docker setup

# Data & Sync
make download       # Download sample data
make sync-data SYNC_HOST=user@brev-instance

# Artifacts
make download-artifacts REMOTE_HOST=user@brev-instance

# Cleanup
make clean          # Clean generated files
```

## Project Structure

```
.
├── configs/          # Configuration files
├── data/              # Data directory
│   ├── raw/           # Raw downloaded data
│   ├── processed/    # Cleaned and tokenized data
│   └── cache/        # Cached teacher predictions
├── scripts/          # Executable scripts
│   ├── data/download.sh    # Data download script
│   └── verify.sh          # Environment verification
├── src/               # Python source code
├── Dockerfile        # Container definition
├── Makefile        # Command management
├── pyproject.toml   # uv project config
└── uv.lock        # Locked dependencies
```

## Requirements

- Python 3.11+
- uv (package manager)
- Docker (optional, for containerized training)
- Teacher model API key (user-provided)
