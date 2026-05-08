"""Simple text dataset loader."""

import json
from pathlib import Path
from typing import Iterator


def load_jsonl(path: str | Path) -> Iterator[dict]:
    """Load data from JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_texts(path: str | Path) -> list[str]:
    """Load texts from JSONL file."""
    texts = []
    for item in load_jsonl(path):
        text = item.get("text", "")
        if text:
            texts.append(text)
    return texts