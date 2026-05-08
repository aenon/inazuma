"""Simple tokenizer wrapper."""

from transformers import PreTrainedTokenizerFast


def load_tokenizer(name: str = "gpt2") -> PreTrainedTokenizerFast:
    """Load tokenizer by name."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(name)


def tokenize_texts(texts: list[str], tokenizer: PreTrainedTokenizerFast, max_length: int = 512):
    """Tokenize texts."""
    return tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )