"""Console entry point for the uv-installed project."""

from src.pipeline import run


def main() -> None:
    """Run the same pipeline as the repository-level main.py script."""
    result = run()

    print("\n========================= DONE =========================")
    print("Initial best equation:")
    print(result["initial"]["model_code"].strip())
    print(f"  validation R^2 = {result['initial']['validation_r2']:.4f}")
    print("\nFinal best equation:")
    print(result["final"]["model_code"].strip())
    print(f"  validation R^2 = {result['final']['validation_r2']:.4f}")
    print(f"\nOutputs written to: {result['output_dir']}")
