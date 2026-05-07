# Minimal Language Model Training Infrastructure - Project Plan

## 1. Project Overview

**Project Name:** Minimal LLM Training Infrastructure  
**Type:** Python-based ML training pipeline  
**Core Functionality:** A minimal yet functional infrastructure for training language models using knowledge distillation from a teacher model, with open data pipelines and uv-based environment management.  
**Target Users:** Developers and researchers who want to train custom language models with limited resources.

---

## 2. Feasibility Analysis

### 2.1 Strengths

- **Proven Technique:** Knowledge distillation (teacher-student) is a well-established method for training smaller models from larger ones
- **Open Data Availability:** Massive open datasets exist (Common Crawl, RedPajama, Pile, etc.)
- **uv Speed:** uv provides 10-100x faster dependency management vs pip
- **Modular Design:** Can start minimal and scale up incrementally
- **Teacher Model Flexibility:** User's API-based teacher model allows GPU-free distillation

### 2.2 Challenges & Mitigations

| Challenge | Mitigation |
|-----------|------------|
| Compute requirements | Use small models (100M-1B params), CPU-friendly options |
| Teacher API costs | Implement caching, use small fine-tuning datasets |
| Data preprocessing complexity | Automate with scripts, use proven tokenizers |
| Storage for datasets | Download-on-demand, cleanup partial files |
| Training stability | Use proven libraries (transformers, deepspeed) |

### 2.3 Risk Assessment

- **LOW:** Core functionality is well-understood, no novel research
- **MEDIUM:** Compute costs depend on usage patterns
- **MEDIUM:** Teacher API dependency - architecture allows switching teachers

### 2.4 Conclusion

✅ **This is a good project** - It addresses a real need with manageable complexity. The minimal approach lowers barriers to entry while remaining extensible.

---

## 3. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    PROJECT STRUCTURE                   │
├─────────────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │    DATA     │───▶│  TEACHER     │───▶│  STUDENT    │  │
│  │  PIPELINE   │    │ INFERENCE   │    │  TRAINING  │  │
│  └─────────────┘    └─────────────┘    └─────────────┘  │
│        │                │                │              │
│        ▼                ▼                ▼              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │  DOWNLOAD  │    │  CACHED    │    │ CHECKPOINT │  │
│  │  STORAGE   │    │ RESPONSES  │    │   STORAGE  │  │
│  └─────────────┘    └─────────────┘    └─────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

1. **Data Pipeline:** Fetch, clean, and tokenize open text data
2. **Teacher Inference:** Generate soft labels via user's API
3. **Student Training:** Train smaller model using distillation loss
4. **Storage:** Manage checkpoints and cached API responses

---

## 4. Implementation Plan

### Phase 1: Environment & Data Pipeline (Week 1)

- [ ] Initialize uv project with pyproject.toml
- [ ] Add core dependencies (transformers, tokenizers, datasets)
- [ ] Create data download scripts for open datasets
- [ ] Implement text cleaning utilities
- [ ] Build tokenization pipeline
- [ ] Create data validation scripts

### Phase 2: Teacher Model Integration (Week 2)

- [ ] Create teacher API client abstraction
- [ ] Implement response caching system
- [ ] Build batch inference pipeline
- [ ] Add retry and error handling
- [ ] Create mock teacher for testing

### Phase 3: Student Model Training (Week 3)

- [ ] Set up distillation training loop
- [ ] Implement loss functions (KL divergence + cross-entropy)
- [ ] Create training config system
- [ ] Add checkpoint management
- [ ] Implement evaluation metrics

### Phase 4: Infrastructure & Documentation (Week 4)

- [ ] Create shell scripts for common operations
- [ ] Add configuration templates
- [ ] Write comprehensive README.md
- [ ] Create quickstart guide
- [ ] Add troubleshooting documentation

---

## 5. Technology Stack

| Component | Technology | Version |
|-----------|------------|---------|
| Package Manager | uv | latest |
| Base Language | Python | 3.11+ |
| Model Framework | transformers | ^4.36 |
| Tokenizer | tokenizers | ^0.15 |
| Dataset Handling | datasets | ^2.16 |
| Training | PyTorch | ^2.1 |
| Config | YAML + python-dotenv | - |

---

## 6. Branch Strategy

- **main:** Production-ready code only
- **feature/*:** Feature development branches
- **dev:** Integration branch before main

### Feature Branch Naming

- `feature/data-pipeline`
- `feature/teacher-inference`
- `feature/student-training`
- `feature/evaluation`
- `feature/documentation`

---

## 7. Success Criteria

- [ ] Can download and preprocess open dataset
- [ ] Can run teacher model inference with caching
- [ ] Can train student model with distillation
- [ ] Can evaluate trained model on benchmarks
- [ ] Documentation enables new user to get started in <30 minutes

---

## 8. Deployment Workflow (Brev GPU)

### 8.1 Overview

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   BUILD      │───▶│   PUSH       │───▶│   VERIFY     │───▶│   TRAIN      │
│  Container   │    │   to Brev    │    │  Environment│    │   Model      │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                                                                    │
                                                                    ▼
                                                           ┌──────────────┐
                                                           │  DOWNLOAD  │
                                                           │ Artifacts  │
                                                           └──────────────┘
```

### 8.2 Docker Image

Create `Dockerfile` with:
- Python 3.11+ base
- uv for package management
- All Python dependencies (no data)
- Project code mounted at runtime

### 8.3 Workflow Steps

| Step | Action | Command/Script |
|------|--------|----------------|
| 1 | Build image | `docker build -t llm-train .` |
| 2 | Push to Brev | `brev push` (or custom registry) |
| 3 | Verify environment | `scripts/verify.sh` (check GPU, deps, connectivity) |
| 4 | Sync data | `rsync/scp` data files to instance |
| 5 | Run training | `uv run python -m src.train.train --config configs/train.yaml` |
| 6 | Download artifacts | `scp` trained model + logs back to local |

### 8.4 Data Sync (Runtime)

The container does NOT contain data. Data is synced at runtime:

```bash
# From local machine to Brev instance
scp -r data/raw/* user@brev-instance:/workspace/project/inazuma/data/raw/
```

### 8.5 Verification Script

Create `scripts/verify.sh` to check:
- [ ] GPU availability (`nvidia-smi`)
- [ ] CUDA working (`python -c "import torch; print(torch.cuda.is_available())"`)
- [ ] Dependencies installed (`uv sync`)
- [ ] Teacher API connectivity
- [ ] Data directory existence
- [ ] Write permissions for outputs

### 8.6 Artifact Download

After training completes:
```bash
# Download from Brev to local
scp -r user@brev-instance:/workspace/project/inazuma/checkpoints/ ./checkpoints/
scp -r user@brev-instance:/workspace/project/inazuma/logs/ ./logs/
```

### 8.7 Reproducibility

- All configs in `configs/` (commit to git)
- Training logs contain all hyperparameters
- checkpoints contain model weights + training state
- Can re-run with same config and resume from checkpoint