"""Simple training script with knowledge distillation."""

import torch
import torch.nn as nn
import yaml
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def load_config(path: str) -> dict:
    """Load training config."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


class DistillationLoss(nn.Module):
    """Knowledge distillation loss."""

    def __init__(self, temperature: float = 2.0, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl = nn.KLDivLoss(reduction="batchmean")
        self.ce = nn.CrossEntropyLoss()

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels: torch.Tensor):
        # Soft target loss
        student_soft = torch.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = torch.softmax(teacher_logits / self.temperature, dim=-1)
        soft_loss = self.kl(student_soft, teacher_soft) * (self.temperature ** 2)

        # Hard target loss
        hard_loss = self.ce(student_logits.view(-1, student_logits.size(-1)), labels.view(-1))

        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss


def train(cfg: dict):
    """Run training."""
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(cfg["student_model"]["name"])
    tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = GPT2LMHeadModel.from_pretrained(cfg["student_model"]["name"])
    model.cuda() if torch.cuda.is_available() else model.cpu()

    # Setup optimizer (use config if valid, otherwise default)
    lr = cfg.get("training", {}).get("learning_rate", 5e-5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr))

    # Simple training loop
    print(f"Starting training with config: {cfg}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("Model loaded successfully!")
    print("Note: Full training loop needs data loader implementation.")

    return model


def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.example.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()