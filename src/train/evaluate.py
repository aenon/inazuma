"""Simple evaluation metrics."""

import math
from collections import Counter

import torch
import torch.nn.functional as F


def perplexity(logits: torch.Tensor, labels: torch.Tensor, pad_token_id: int = 50256) -> float:
    """Calculate perplexity."""
    # Simple version
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=pad_token_id)
    return math.exp(loss.item())


def accuracy(logits: torch.Tensor, labels: torch.Tensor, pad_token_id: int = 50256) -> float:
    """Calculate token accuracy."""
    preds = logits.argmax(dim=-1)
    mask = labels != pad_token_id
    correct = (preds == labels) & mask
    return correct.sum().item() / mask.sum().item()


def compute_metrics(model, batch, device: str = "cpu") -> dict:
    """Compute evaluation metrics."""
    model.eval()
    with torch.no_grad():
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids)
        logits = outputs.logits
    return {
        "perplexity": perplexity(logits, labels),
        "accuracy": accuracy(logits, labels),
    }