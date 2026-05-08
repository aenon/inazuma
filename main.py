"""Main entry point for LLM training infrastructure."""

import sys


def main():
    """Main entry point."""
    print("Minimal LLM Training Infrastructure")
    print("")
    print("Usage:")
    print("  make train                      # Run training")
    print("  uv run python -m src.train.train --help  # Training options")
    print("  make verify                     # Verify environment")
    sys.exit(1)


if __name__ == "__main__":
    main()
