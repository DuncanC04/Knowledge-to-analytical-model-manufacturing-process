#!/usr/bin/env python
"""Knowledge-to-Equation — command-line entry point.

Runs the full pipeline (knowledge retrieval -> initial equations -> refinement)
using settings from config/config.yaml, and writes the result text files into
the configured output directory.

Usage:
    python main.py                    # use config/config.yaml
    python main.py --config my.yaml   # use a custom config file
"""

import argparse

import nest_asyncio

from src.pipeline import run


def main():
    parser = argparse.ArgumentParser(description="Knowledge-to-Equation pipeline")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to the config YAML (defaults to config/config.yaml).",
    )
    args = parser.parse_args()

    # Allow nested event loops (LlamaParse / LlamaIndex use asyncio internally).
    nest_asyncio.apply()

    result = run(config_path=args.config)

    print("\n========================= DONE =========================")
    print("Initial best equation:")
    print(result["initial"]["model_code"].strip())
    print(f"  validation R^2 = {result['initial']['validation_r2']:.4f}")
    print("\nFinal best equation:")
    print(result["final"]["model_code"].strip())
    print(f"  validation R^2 = {result['final']['validation_r2']:.4f}")
    print(f"\nOutputs written to: {result['output_dir']}")


if __name__ == "__main__":
    main()
