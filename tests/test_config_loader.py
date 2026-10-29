import tempfile
import unittest
from pathlib import Path

from src.config_loader import ConfigError, load_config, variable_phrase


class ConfigLoaderTests(unittest.TestCase):
    def test_loads_active_and_all_outputs(self):
        cfg = load_config()
        self.assertIn(cfg["output_column"], cfg["output_columns"])
        self.assertEqual(len(cfg["input_columns"]), 4)

    def test_rejects_duplicate_input_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text(
                "variables:\n"
                "  inputs:\n"
                "    - {name: A, symbol: A, column: same}\n"
                "    - {name: B, symbol: B, column: same}\n"
                "  output: {name: Y, symbol: Y, column: y}\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigError, "input columns"):
                load_config(path)

    def test_variable_phrase_requires_variables(self):
        with self.assertRaises(ValueError):
            variable_phrase([])
