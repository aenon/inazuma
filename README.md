# Minimal Language Model Training Infrastructure

A minimal yet functional infrastructure for training language models using knowledge distillation from a teacher model.

## Quick Start

```bash
# 1. Initialize uv environment
uv venv
uv sync

# 2. Configure teacher API
cp configs/teacher.example.env configs/teacher.env
# Edit configs/teacher.env with your API credentials

# 3. Download sample data
uv run python scripts/data/download.py --dataset sample

# 4. Generate teacher predictions
uv run python scripts/teacher/inference.py --data data/raw/sample.jsonl

# 5. Train student model
uv run python scripts/train/train.py --config configs/train.yaml
```

## Project Structure

```
├── configs/          # Configuration files
├── data/              # Data directory (downloads here)
│   ├── raw/           # Raw downloaded data
│   ├── processed/    # Cleaned and tokenized data
│   └── cache/        # Cached teacher predictions
├── scripts/          # Executable scripts
│   ├── data/         # Data download/preprocess
│   ├── teacher/      # Teacher inference
│   └── train/        # Training scripts
├── src/               # Python source code
└── docs/              # Documentation
```

## Requirements

- Python 3.11+
- uv (package manager)
- Teacher model API key (user-provided)

## Documentation

See `docs/` directory for detailed guides.
