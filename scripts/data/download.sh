#!/bin/bash
# Data download script

set -e

DATASET="${1:-sample}"
OUTPUT_DIR="${2:-data/raw}"

echo "Downloading dataset: $DATASET"

case "$DATASET" in
    sample)
        echo "Creating sample data..."
        mkdir -p "$OUTPUT_DIR"
        echo '{"text": "The quick brown fox jumps over the lazy dog."}' > "$OUTPUT_DIR/sample.jsonl"
        echo '{"text": "A journey of a thousand miles begins with a single step."}' >> "$OUTPUT_DIR/sample.jsonl"
        echo '{"text": "Knowledge is power."}' >> "$OUTPUT_DIR/sample.jsonl"
        ;;
    *)
        echo "Unknown dataset: $DATASET"
        echo "Available: sample"
        exit 1
        ;;
esac

echo "Done! Data saved to $OUTPUT_DIR"