"""Load the YAML config and the API keys from config/.env."""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Project root = parent of the `src` directory.
ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: str | os.PathLike | None = None) -> dict:
    """Read config.yaml and load the API keys from config/.env into the env.

    Returns the parsed config as a dict, with a few convenience helpers added:
      cfg["_root"]           -> project root Path
      cfg["input_symbols"]   -> list of input symbols, in order
      cfg["input_columns"]   -> list of input column names, in order
      cfg["output_symbol"]   -> output symbol
      cfg["output_column"]   -> output column name
    """
    config_path = Path(config_path) if config_path else ROOT / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Load API keys from config/.env (falls back to any keys already in the env).
    load_dotenv(ROOT / "config" / ".env")

    cfg["_root"] = ROOT
    inputs = cfg["variables"]["inputs"]
    output = cfg["variables"]["output"]
    cfg["input_symbols"] = [v["symbol"] for v in inputs]
    cfg["input_columns"] = [v["column"] for v in inputs]
    cfg["output_symbol"] = output["symbol"]
    cfg["output_column"] = output["column"]
    return cfg


def require_env(name: str) -> str:
    """Return an environment variable or raise a clear error if it is missing."""
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"Missing API key '{name}'. Add it to config/.env "
            f"(see config/.env.example)."
        )
    return value


def resolve_path(cfg: dict, path: str) -> Path:
    """Resolve a (possibly relative) config path against the project root."""
    p = Path(path)
    return p if p.is_absolute() else cfg["_root"] / p


def variable_phrase(variables: list[dict]) -> str:
    """Render a list of variables as 'Name (SYM), Name (SYM), ...' for prompts."""
    return ", ".join(f"{v['name']} ({v['symbol']})" for v in variables)
