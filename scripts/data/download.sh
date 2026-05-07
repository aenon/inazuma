#!/bin/bash
# Data download script for open datasets

set -e

DATASET="${1:-sample}"
OUTPUT_DIR="${2:-data/raw}"

echo "Downloading dataset: $DATASET"

mkdir -p "$OUTPUT_DIR"

case "$DATASET" in
    sample)
        echo "Creating sample data..."
        echo '{"text": "The quick brown fox jumps over the lazy dog."}' > "$OUTPUT_DIR/sample.jsonl"
        echo '{"text": "A journey of a thousand miles begins with a single step."}' >> "$OUTPUT_DIR/sample.jsonl"
        echo '{"text": "Knowledge is power."}' >> "$OUTPUT_DIR/sample.jsonl"
        ;;

    tiny)
        echo "Downloading TinyStories (small subset)..."
        uv run python -c "
from datasets import load_dataset
import json

# Load tiny version for testing
ds = load_dataset('roneneldan/TinyStories', split='train[:1000]')
print(f'Loaded {len(ds)} examples')

with open('$OUTPUT_DIR/tiny.jsonl', 'w') as f:
    for item in ds:
        f.write(json.dumps({'text': item['text']}) + '\n')
print('Saved to $OUTPUT_DIR/tiny.jsonl')
"
        ;;

    wikitext)
        echo "Downloading WikiText-2-raw (subset)..."
        uv run python -c "
from datasets import load_dataset
import json

ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train[:2000]')
print(f'Loaded {len(ds)} examples')

# Filter non-empty
texts = [item['text'] for item in ds if item['text'].strip()]

with open('$OUTPUT_DIR/wikitext.jsonl', 'w') as f:
    for text in texts[:500]:
        f.write(json.dumps({'text': text}) + '\n')
print(f'Saved {min(500, len(texts))} examples to $OUTPUT_DIR/wikitext.jsonl')
"
        ;;

    python-code)
        echo "Downloading CodeSearchNet Python (software engineering)..."
        uv run python -c "
from datasets import load_dataset
import json

# Load CodeSearchNet Python dataset
ds = load_dataset('code_search_net', 'python', split='train')
print(f'Loaded {len(ds)} examples')

# Convert to JSONL with text field
with open('$OUTPUT_DIR/python-code.jsonl', 'w') as f:
    for item in ds:
        # Each item has 'whole_func_string' - the complete function code
        code = item.get('whole_func_string', '')
        if code.strip():
            f.write(json.dumps({'text': code}) + '\n')
print(f'Saved {len(ds)} examples to $OUTPUT_DIR/python-code.jsonl')
print(f'Approx size: {sum(len(x.get(\"whole_func_string\", \"\")) / 1024 / 1024 for x in ds):.1f} MB')
"
        ;;

    *)
        echo "Unknown dataset: $DATASET"
        echo "Available: sample, tiny, wikitext, python-code"
        exit 1
        ;;
esac

echo "Done! Data saved to $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR/"