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
    from src.data import load_texts
    from tqdm import tqdm
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(cfg["student_model"]["name"])
    tokenizer.pad_token = tokenizer.eos_token

    # Load student model (trainable)
    student = GPT2LMHeadModel.from_pretrained(cfg["student_model"]["name"])
    student.cuda() if torch.cuda.is_available() else student.cpu()
    
    # Load teacher model (frozen)
    teacher = GPT2LMHeadModel.from_pretrained(cfg["student_model"]["name"])
    teacher.cuda() if torch.cuda.is_available() else teacher.cpu()
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Training config
    batch_size = cfg.get("training", {}).get("batch_size", 4)
    max_seq_length = cfg.get("training", {}).get("max_seq_length", 256)
    num_epochs = cfg.get("training", {}).get("num_epochs", 1)
    learning_rate = float(cfg.get("training", {}).get("learning_rate", "5e-5"))
    
    # Data path
    data_path = cfg.get("data", {}).get("train_path", "data/raw/python-code.jsonl")
    print(f"Loading data from: {data_path}")
    
    # Limit for quick test
    max_examples = 20  # Quick test with 20 examples
    
    texts = load_texts(data_path)[:max_examples]
    print(f"Loaded {len(texts)} examples")
    
    # Tokenize
    print("Tokenizing...")
    encodings = tokenizer(texts, truncation=True, max_length=max_seq_length, 
                        padding="max_length", return_tensors="pt")
    
    # Optimizer
    optimizer = torch.optim.AdamW(student.parameters(), lr=learning_rate)
    criterion = nn.KLDivLoss(reduction="batchmean")
    
    # Training loop
    print(f"\nStarting self-distillation training...")
    print(f"Batch size: {batch_size}, Epochs: {num_epochs}, LR: {learning_rate}")
    
    student.train()
    total_steps = 0
    
    for epoch in range(num_epochs):
        for batch_start in range(0, len(texts), batch_size):
            batch_end = min(batch_start + batch_size, len(texts))
            
            input_ids = encodings["input_ids"][batch_start:batch_end].to(student.device)
            attention_mask = encodings["attention_mask"][batch_start:batch_end].to(student.device)
            
            # Get teacher logits (no grad)
            with torch.no_grad():
                teacher_outputs = teacher(input_ids=input_ids, attention_mask=attention_mask)
                teacher_logits = teacher_outputs.logits / cfg.get("distillation", {}).get("temperature", 2.0)
            
            # Get student logits
            student_outputs = student(input_ids=input_ids, attention_mask=attention_mask)
            student_logits = student_outputs.logits / cfg.get("distillation", {}).get("temperature", 2.0)
            
            # KL divergence loss
            loss = criterion(
                torch.log_softmax(student_logits, dim=-1),
                torch.softmax(teacher_logits, dim=-1)
            )
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_steps += 1
            
            if total_steps % 20 == 0:
                print(f"Step {total_steps}, Loss: {loss.item():.4f}")
    
    print(f"\nTraining complete! {total_steps} steps")
    return student


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