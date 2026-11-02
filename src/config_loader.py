"""Load the YAML config and the API keys from config/.env."""

import os
from pathlib import Path
from collections.abc import Mapping, Sequence

import yaml
from dotenv import load_dotenv

# Project root = parent of the `src` directory.
ROOT = Path(__file__).resolve().parent.parent


class ConfigError(ValueError):
    """Raised when the research configuration violates its input contract."""


def _mapping(value: object, name: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{name} must be a mapping")
    return value


def _required_text(item: Mapping, key: str, name: str) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name}.{key} must be a non-empty string")
    return value.strip()


def _validate_variables(cfg: Mapping) -> tuple[list[dict], list[dict]]:
    variables = _mapping(cfg.get("variables"), "variables")
    raw_inputs = variables.get("inputs")
    if not isinstance(raw_inputs, Sequence) or isinstance(raw_inputs, (str, bytes)) or not raw_inputs:
        raise ConfigError("variables.inputs must be a non-empty list")

    inputs: list[dict] = []
    for index, raw in enumerate(raw_inputs):
        item = _mapping(raw, f"variables.inputs[{index}]")
        inputs.append(
            {
                "name": _required_text(item, "name", f"variables.inputs[{index}]"),
                "symbol": _required_text(item, "symbol", f"variables.inputs[{index}]"),
                "column": _required_text(item, "column", f"variables.inputs[{index}]"),
            }
        )

    raw_output = _mapping(variables.get("output"), "variables.output")
    active_output = {
        "name": _required_text(raw_output, "name", "variables.output"),
        "symbol": _required_text(raw_output, "symbol", "variables.output"),
        "column": _required_text(raw_output, "column", "variables.output"),
    }
    raw_outputs = variables.get("outputs", [active_output])
    if not isinstance(raw_outputs, Sequence) or isinstance(raw_outputs, (str, bytes)) or not raw_outputs:
        raise ConfigError("variables.outputs must be a non-empty list")
    outputs: list[dict] = []
    for index, raw in enumerate(raw_outputs):
        item = _mapping(raw, f"variables.outputs[{index}]")
        outputs.append(
            {
                "name": _required_text(item, "name", f"variables.outputs[{index}]"),
                "symbol": _required_text(item, "symbol", f"variables.outputs[{index}]"),
                "column": _required_text(item, "column", f"variables.outputs[{index}]"),
            }
        )

    input_symbols = [item["symbol"] for item in inputs]
    input_columns = [item["column"] for item in inputs]
    output_symbols = [item["symbol"] for item in outputs]
    output_columns = [item["column"] for item in outputs]
    for label, values in (
        ("input symbols", input_symbols),
        ("input columns", input_columns),
        ("output symbols", output_symbols),
        ("output columns", output_columns),
    ):
        if len(values) != len(set(values)):
            raise ConfigError(f"{label} must be unique")
    if active_output["column"] not in output_columns:
        raise ConfigError("variables.output must reference one of variables.outputs")
    if set(input_columns) & set(output_columns):
        raise ConfigError("input and output columns must be disjoint")
    return inputs, outputs


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
    config_path = config_path.resolve()
    try:
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except OSError as exc:
        raise ConfigError(f"Could not read configuration: {config_path}") from exc
    if not isinstance(cfg, dict):
        raise ConfigError("The configuration root must be a YAML mapping")

    inputs, outputs = _validate_variables(cfg)

    # Load API keys from config/.env (falls back to any keys already in the env).
    # The project keeps its active credentials in config/.env. Override stale
    # process-level values so replacing a key in that file actually affects
    # CLI runs launched through uv or an IDE terminal.
    load_dotenv(ROOT / "config" / ".env", override=True)

    cfg["_root"] = ROOT
    output = cfg["variables"]["output"]
    cfg["variables"]["inputs"] = inputs
    cfg["variables"]["outputs"] = outputs
    cfg["input_symbols"] = [v["symbol"] for v in inputs]
    cfg["input_columns"] = [v["column"] for v in inputs]
    cfg["output_symbol"] = output["symbol"]
    cfg["output_column"] = output["column"]
    cfg["output_columns"] = [item["column"] for item in outputs]
    cfg["_config_path"] = config_path
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


def resolve_path(cfg: Mapping, path: str | os.PathLike) -> Path:
    """Resolve a (possibly relative) config path against the project root."""
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError("path must be a string or path-like value")
    p = Path(path)
    return p if p.is_absolute() else cfg["_root"] / p


def variable_phrase(variables: Sequence[Mapping]) -> str:
    """Render a list of variables as 'Name (SYM), Name (SYM), ...' for prompts."""
    if not variables:
        raise ValueError("variables must contain at least one item")
    return ", ".join(f"{v['name']} ({v['symbol']})" for v in variables)
