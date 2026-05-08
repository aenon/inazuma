# Minimal LLM Training Infrastructure - Makefile

# Configuration
PYTHON := python3
UV := uv
DOCKER := docker
CONFIG ?= configs/train.example.yaml
TEACHER_CONFIG ?= configs/teacher.env

# Colors
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m

.PHONY: help install setup verify build run-test train clean docker-build docker-verify sync-data download-artifacts

# Default target
help:
	@echo "$(GREEN)Minimal LLM Training Infrastructure$(NC)"
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@echo "$(YELLOW)Setup & Installation$(NC)"
	@echo "  install         Install dependencies (uv sync)"
	@echo "  setup          Full setup (install + download data)"
	@echo "  verify         Verify environment is ready"
	@echo ""
	@echo "$(YELLOW)Training$(NC)"
	@echo "  train          Run training with CONFIG=$(NC)"
	@echo "  run-test       Quick test run"
	@echo ""
	@echo "$(YELLOW)Data$(NC)"
	@echo "  download      Download sample dataset"
	@echo "  sync-data     Sync data to remote (SYNC_HOST=user@host)"
	@echo ""
	@echo "$(YELLOW)Docker$(NC)"
	@echo "  docker-build Build Docker image"
	@echo "  docker-verify Verify Docker setup"
	@echo ""
	@echo "$(YELLOW)Artifacts$(NC)"
	@echo "  download-artifacts Download trained model from remote"
	@echo ""
	@echo "$(YELLOW)Cleanup$(NC)"
	@echo "  clean          Clean generated files"
	@echo ""
	@echo "Examples:"
	@echo "  make train CONFIG=configs/train.yaml"
	@echo "  make sync-data SYNC_HOST=user@brev-instance"
	@echo "  make docker-build"

install:
	@echo "Installing dependencies..."
	$(UV) sync

setup: install download verify
	@echo "$(GREEN)Setup complete!$(NC)"

verify:
	@echo "Verifying environment..."
	@bash scripts/verify.sh

download:
	@echo "Downloading sample data..."
	@bash scripts/data/download.sh sample data/raw

train:
	@$(UV) run python -m src.train.train --config $(CONFIG)

run-test:
	@echo "Running test (placeholder)..."
	@echo "Implement training script to enable this target"

docker-build:
	@echo "Building Docker image..."
	$(DOCKER) build -t llm-train .

docker-verify:
	@echo "Verifying Docker image..."
	$(DOCKER) run --rm --gpus all llm-train bash scripts/verify.sh

sync-data:
ifndef SYNC_HOST
	@echo "$(YELLOW)Error: SYNC_HOST required$(NC)"
	@echo "Usage: make sync-data SYNC_HOST=user@brev-instance"
	@exit 1
endif
	@echo "Syncing data to $(SYNC_HOST)..."
	scp -r data/raw/* $(SYNC_HOST):/workspace/project/inazuma/data/raw/

download-artifacts:
ifndef REMOTE_HOST
	@echo "$(YELLOW)Error: REMOTE_HOST required$(NC)"
	@echo "Usage: make download-artifacts REMOTE_HOST=user@brev-instance"
	@exit 1
endif
	@echo "Downloading artifacts from $(REMOTE_HOST)..."
	scp -r $(REMOTE_HOST):/workspace/project/inazuma/checkpoints/ ./checkpoints/
	scp -r $(REMOTE_HOST):/workspace/project/inazuma/logs/ ./logs/

clean:
	@echo "Cleaning generated files..."
	rm -rf checkpoints/* logs/* data/processed/* data/cache/*
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)Clean complete!$(NC)"