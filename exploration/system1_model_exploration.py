"""Exploration scaffold for System 1 relevance filtering.

This is separate from the comparison pipeline. It reuses the existing
Docling-based PDF parsing and cache, but leaves the System 1 model to be
implemented as part of the exploration.

Run from the repository root:
    uv run python exploration/system1_model_exploration.py
"""

from __future__ import annotations

from typing import Any

from src.config_loader import load_config
from src.knowledge import _load_documents, _local_config

from laya import Router


def load_parsed_documents(config_path: str | None = None) -> list[Any]:
    """Parse/cache configured PDFs and return them as LlamaIndex documents."""
    cfg = load_config(config_path)
    return _load_documents(cfg, _local_config(cfg))


def filter_chunk_with_system1(chunk: Any) -> Any:
    """Apply System 1 filtering to one parsed chunk."""
    router = Router()

    state = chunk

    questions = {
        "physical_relationship": {
            "type": "noul",
            "instructions": (
                "Does this passage explicitly describe a relationship between "
                "two or more physical variables, process parameters, or measured "
                "outcomes? Include qualitative trends, quantitative relationships, "
                "correlations, and conditional effects. Do not count simple "
                "co-occurrence of variable names."
            )
        },

        "mathematical_relationship": {
            "type": "noul",
            "instructions": (
                "Does this passage contain or describe a mathematical relationship "
                "between physical quantities, such as a governing equation, "
                "empirical formula, proportionality, or scaling law? "
                "Exclude unrelated mathematical expressions."
            )
        },

        "physical_mechanism": {
            "type": "noul",
            "instructions": (
                "Does this passage explain how or why a physical process occurs, "
                "including interactions between variables, underlying physical "
                "principles, or intermediate mechanisms connecting process "
                "parameters to measured outcomes?"
            )
        },

        "quantitative_evidence": {
            "type": "noul",
            "instructions": (
                "Does this passage report numerical evidence about a relationship "
                "between physical quantities, such as measured changes, numerical "
                "coefficients, fitted parameters, or quantified trends? "
                "Do not count isolated experimental settings or parameter values."
            )
        },

        "operating_conditions": {
            "type": "noul",
            "instructions": (
                "Does this passage specify experimental parameter values, "
                "operating ranges, fixed conditions, material properties, "
                "or boundary conditions that could help determine where "
                "a physical relationship or predictive model is applicable?"
            )
        }
    }

    result = router.predict(state, questions)
    return(result)



def main() -> None:
    """Verify PDF parsing and caching before adding the System 1 model."""
    documents = load_parsed_documents()
    print(f"Parsed or loaded {len(documents)} PDF document(s):")
    for document in documents:
        source = document.metadata.get("file_name", "unknown source")
        print(f"- {source}: {len(document.text)} characters")


if __name__ == "__main__":
    main()
