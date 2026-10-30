"""Run Laya System 1 labels over the parsed literature corpus.

The Laya ``noul`` probability is interpreted explicitly: ``noul > threshold``
is a positive/yes label. This module keeps the raw answer and derived decision
together so thresholding is auditable.

Run from the repository root::

    uv run python exploration/system1_model_exploration.py
"""

from __future__ import annotations

import json
import re
import argparse
from dataclasses import asdict, dataclass, replace
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import openai
from laya import Router
from dotenv import load_dotenv
from llama_index.core.node_parser import SentenceSplitter

# Support direct execution with ``python exploration/system1_model_exploration.py``.
# In that mode Python puts ``exploration/`` on sys.path, not the repository root.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.config_loader import load_config, require_env
from src.knowledge import _load_documents, _local_config


LABEL_QUESTIONS: dict[str, dict[str, str]] = {
    "physical_relationship": {
        "type": "noul",
        "instructions": (
            "Does this passage explicitly describe a relationship between "
            "two or more physical variables, process parameters, or measured "
            "outcomes? Include qualitative trends, quantitative relationships, "
            "correlations, and conditional effects. Do not count simple "
            "co-occurrence of variable names."
        ),
    },
    "mathematical_relationship": {
        "type": "noul",
        "instructions": (
            "Does this passage contain or describe a mathematical relationship "
            "between physical quantities, such as a governing equation, "
            "empirical formula, proportionality, or scaling law? "
            "Exclude unrelated mathematical expressions."
        ),
    },
    "physical_mechanism": {
        "type": "noul",
        "instructions": (
            "Does this passage explain how or why a physical process occurs, "
            "including interactions between variables, underlying physical "
            "principles, or intermediate mechanisms connecting process "
            "parameters to measured outcomes?"
        ),
    },
    "quantitative_evidence": {
        "type": "noul",
        "instructions": (
            "Does this passage report numerical evidence about a relationship "
            "between physical quantities, such as measured changes, numerical "
            "coefficients, fitted parameters, or quantified trends? "
            "Do not count isolated experimental settings or parameter values."
        ),
    },
    "operating_conditions": {
        "type": "noul",
        "instructions": (
            "Does this passage specify experimental parameter values, "
            "operating ranges, fixed conditions, material properties, "
            "or boundary conditions that could help determine where "
            "a physical relationship or predictive model is applicable?"
        ),
    },
}

# Laya's local model returns typed classifications, not free-form text. These
# questions therefore capture the relationship's structure and preserve the
# source passage as the evidence for the exact wording and variable names.
RELATIONSHIP_QUESTIONS: dict[str, dict[str, Any]] = {
    "relationship_form": {
        "type": "choice",
        "instructions": "What form does the relationship take in this passage?",
        "criteria": {
            "equation_or_formula": None,
            "empirical_fit_or_model": None,
            "qualitative_trend": None,
            "correlation_or_association": None,
            "causal_or_mechanistic": None,
            "comparative_effect": None,
            "threshold_or_nonlinear": None,
            "not_stated": None,
        },
    },
    "relationship_direction": {
        "type": "choice",
        "instructions": "What direction or behavior is reported for the relationship?",
        "criteria": {
            "increases": None,
            "decreases": None,
            "inverse": None,
            "nonlinear_or_nonmonotonic": None,
            "conditional_or_interaction": None,
            "no_direction_stated": None,
            "not_stated": None,
        },
    },
    "evidence_basis": {
        "type": "choice",
        "instructions": "What is the evidence basis for the relationship?",
        "criteria": {
            "measured_experiment": None,
            "fitted_from_data": None,
            "theoretical_or_derived": None,
            "simulation_or_model": None,
            "mixed": None,
            "not_stated": None,
        },
    },
    "range_present": {
        "type": "noul",
        "instructions": (
            "Does the passage state an explicit tested or applicable range, "
            "minimum, maximum, interval, or boundary for a variable?"
        ),
    },
    "fixed_conditions_present": {
        "type": "noul",
        "instructions": (
            "Does the passage state fixed operating conditions or controlled "
            "parameters under which the relationship was observed?"
        ),
    },
    "applicability_limit_present": {
        "type": "noul",
        "instructions": (
            "Does the passage state a limit, threshold, transition, or caveat "
            "for when the relationship applies?"
        ),
    },
}


@dataclass(frozen=True)
class LabelDecision:
    """Auditable, JSON-serializable result for one passage."""

    passage_id: str
    evidence_text: str
    range_mentions: list[str]
    score: float
    retained: bool
    labels: dict[str, bool]
    noul_probabilities: dict[str, float]
    usage: dict[str, Any]
    raw_answers: dict[str, Any]
    relationship_facts: dict[str, Any]
    range_facts: dict[str, Any]
    extraction_answers: dict[str, Any]
    llm_description: dict[str, Any] | None


def load_parsed_documents(config_path: str | None = None) -> list[Any]:
    """Parse/cache configured PDFs and return them as LlamaIndex documents."""
    cfg = load_config(config_path)
    return _load_documents(cfg, _local_config(cfg))


def chunk_documents(
    documents: list[Any],
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[Any]:
    """Split each paper into sentence-aware chunks before System 1 labeling."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be nonnegative and smaller than chunk_size")

    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.get_nodes_from_documents(documents)
    for index, chunk in enumerate(chunks):
        source = chunk.metadata.get("file_name", "unknown_source")
        chunk.metadata["passage_id"] = f"{source}:chunk-{index}"
        chunk.metadata["chunk_index"] = index
    return chunks


def create_router(device: str = "cuda") -> Router:
    """Load Laya on the requested device and fail instead of silently using CPU."""
    router = Router(device=device)
    actual_device = str(getattr(router, "device", ""))
    if device.startswith("cuda") and not actual_device.startswith("cuda"):
        raise RuntimeError(
            f"Laya did not start on GPU: requested {device!r}, got {actual_device!r}. "
            "Check the CUDA-enabled PyTorch installation and GPU visibility."
        )
    return router


def _get(value: Any, key: str, default: Any = None) -> Any:
    """Read a field from either a mapping or a Laya response object."""
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _jsonable(value: Any) -> Any:
    """Convert common Laya response objects into JSON-compatible values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    return str(value)


def _passage_id(chunk: Any, fallback: str) -> str:
    metadata = _get(chunk, "metadata", {}) or {}
    if isinstance(metadata, Mapping):
        return str(metadata.get("passage_id") or metadata.get("id") or fallback)
    return fallback


def _evidence_text(chunk: Any) -> str:
    """Keep the source text because Laya classifications do not quote text."""
    text = _get(chunk, "text")
    if text is None and hasattr(chunk, "get_content"):
        text = chunk.get_content()
    return str(text or "")


def _router_state(chunk: Any) -> str:
    """Return the serializable passage text expected by Laya."""
    text = _evidence_text(chunk)
    if not text.strip():
        raise ValueError("Cannot run Laya on a passage with no text")
    # Laya's checkpoint has an 8,192-token maximum. Keep enough text for the
    # classifier while retaining the complete passage separately in the result.
    max_chars = 18_000
    if len(text) <= max_chars:
        return text
    head = max_chars * 2 // 3
    tail = max_chars - head
    return text[:head] + "\n[...middle omitted for Laya... ]\n" + text[-tail:]


def _range_mentions(text: str) -> list[str]:
    """Extract numeric range phrases for review; this is not a model judgment."""
    pattern = re.compile(
        r"\b(?:between\s+|from\s+)?\d+(?:\.\d+)?\s*(?:to|-|–|—)\s*"
        r"\d+(?:\.\d+)?\s*[A-Za-z%°µμ/\-^0-9]*"
    )
    return list(dict.fromkeys(match.group(0).strip() for match in pattern.finditer(text)))


def _extract_relationship_facts(
    chunk: Any,
    *,
    router: Router,
    threshold: float,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Ask structured follow-up questions for a positively labeled passage."""
    result = router.predict(_router_state(chunk), RELATIONSHIP_QUESTIONS)
    answers = _get(result, "answers", {}) or {}
    extraction_answers = _jsonable(answers)
    if not isinstance(extraction_answers, dict):
        extraction_answers = {}

    relationship_facts: dict[str, Any] = {}
    range_facts: dict[str, Any] = {}
    for name, answer in extraction_answers.items():
        if name in {"relationship_form", "relationship_direction", "evidence_basis"}:
            relationship_facts[name] = answer.get("choice") if isinstance(answer, dict) else answer
        elif name in {"range_present", "fixed_conditions_present", "applicability_limit_present"}:
            probability = float(answer.get("noul", 0.0)) if isinstance(answer, dict) else 0.0
            range_facts[name] = {
                "noul_probability": probability,
                "yes": probability > threshold,
            }
    return relationship_facts, range_facts, extraction_answers


def classify_chunk(
    chunk: Any,
    *,
    router: Router | None = None,
    threshold: float = 0.5,
    min_positive_labels: int = 1,
    exclude_truncated: bool = False,
    fallback_id: str = "unknown",
) -> LabelDecision:
    """Classify one chunk using ``noul > threshold`` as the yes decision.

    ``score`` is the fraction of requested labels classified positive. A
    passage is retained when it meets ``min_positive_labels``. Truncated
    requests are retained by default but recorded in ``usage``; callers can
    set ``exclude_truncated=True`` for a conservative pass.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if min_positive_labels < 1 or min_positive_labels > len(LABEL_QUESTIONS):
        raise ValueError("min_positive_labels is outside the label range")

    active_router = router or create_router()
    raw_result = active_router.predict(_router_state(chunk), LABEL_QUESTIONS)
    answers = _get(raw_result, "answers", {}) or {}
    usage = _jsonable(_get(raw_result, "usage", {}) or {})
    if not isinstance(usage, dict):
        usage = {}

    labels: dict[str, bool] = {}
    probabilities: dict[str, float] = {}
    raw_answers: dict[str, Any] = {}
    for name in LABEL_QUESTIONS:
        answer = _get(answers, name, {}) or {}
        probability = _get(answer, "noul")
        if probability is None:
            raise ValueError(f"Laya response has no noul probability for {name}")
        probability = float(probability)
        probabilities[name] = probability
        labels[name] = probability > threshold
        raw_answers[name] = _jsonable(answer)

    positive_count = sum(labels.values())
    score = positive_count / len(LABEL_QUESTIONS)
    truncated = bool(usage.get("truncated", False))
    retained = positive_count >= min_positive_labels and not (
        exclude_truncated and truncated
    )
    usage["threshold"] = threshold
    usage["positive_count"] = positive_count
    usage["truncated_excluded"] = exclude_truncated and truncated

    relationship_facts: dict[str, Any] = {}
    range_facts: dict[str, Any] = {}
    extraction_answers: dict[str, Any] = {}
    if retained:
        (
            relationship_facts,
            range_facts,
            extraction_answers,
        ) = _extract_relationship_facts(
            chunk,
            router=active_router,
            threshold=threshold,
        )

    evidence_text = _evidence_text(chunk)
    return LabelDecision(
        passage_id=_passage_id(chunk, fallback_id),
        evidence_text=evidence_text,
        range_mentions=_range_mentions(evidence_text),
        score=score,
        retained=retained,
        labels=labels,
        noul_probabilities=probabilities,
        usage=usage,
        raw_answers=raw_answers,
        relationship_facts=relationship_facts,
        range_facts=range_facts,
        extraction_answers=extraction_answers,
        llm_description=None,
    )


def describe_with_llm(
    decision: LabelDecision,
    *,
    client: openai.OpenAI,
    model: str,
    max_tokens: int = 300,
    temperature: float = 0.1,
) -> dict[str, Any]:
    """Generate a concise evidence-grounded description after Laya screening."""
    if not decision.retained:
        raise ValueError("LLM descriptions are only generated for retained passages")

    context = {
        "passage_id": decision.passage_id,
        "laya_labels": decision.labels,
        "laya_relationship_facts": decision.relationship_facts,
        "laya_range_facts": decision.range_facts,
        "candidate_numeric_ranges": decision.range_mentions,
        "evidence_text": decision.evidence_text[:16000],
    }
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You extract concise, conservative scientific evidence from "
                    "manufacturing literature. Laya's classifications are hints, "
                    "not facts. Use only the supplied passage. Do not invent "
                    "variables, directions, equations, values, units, or ranges. "
                    "Return valid JSON with exactly these keys: summary, "
                    "variables, relationship, mechanism, ranges, conditions, "
                    "evidence_quote. Use empty strings or empty lists when the "
                    "passage does not support a field. Keep summary under 80 words."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(context, ensure_ascii=False),
            },
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or "{}"
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = {"summary": content, "parse_error": True}
    if not isinstance(parsed, dict):
        parsed = {"summary": str(parsed), "parse_error": True}
    parsed["model"] = model
    return parsed


def add_llm_descriptions(
    decisions: list[LabelDecision],
    *,
    client: openai.OpenAI,
    model: str,
    max_tokens: int = 300,
    temperature: float = 0.1,
) -> list[LabelDecision]:
    """Run the second-stage generator only on Laya-retained passages."""
    described: list[LabelDecision] = []
    for decision in decisions:
        if decision.retained:
            description = describe_with_llm(
                decision,
                client=client,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            described.append(replace(decision, llm_description=description))
        else:
            described.append(decision)
    return described


def filter_chunk_with_system1(
    chunk: Any,
    *,
    threshold: float = 0.5,
    min_positive_labels: int = 1,
    exclude_truncated: bool = False,
) -> LabelDecision:
    """Backward-compatible entry point for filtering one parsed chunk."""
    return classify_chunk(
        chunk,
        threshold=threshold,
        min_positive_labels=min_positive_labels,
        exclude_truncated=exclude_truncated,
    )


def filter_documents(
    documents: list[Any],
    *,
    threshold: float = 0.5,
    min_positive_labels: int = 1,
    exclude_truncated: bool = False,
) -> list[LabelDecision]:
    """Classify every chunk and return auditable decisions."""
    router = create_router()
    return [
        classify_chunk(
            document,
            router=router,
            threshold=threshold,
            min_positive_labels=min_positive_labels,
            exclude_truncated=exclude_truncated,
            fallback_id=f"chunk-{index}",
        )
        for index, document in enumerate(documents)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Optional YAML config path.")
    parser.add_argument(
        "--skip-descriptions",
        action="store_true",
        help="Run only Laya screening and structured extraction.",
    )
    parser.add_argument("--model", default=None, help="Override the configured LLM.")
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument(
        "--output",
        default=str(REPOSITORY_ROOT / "output" / "system1_labels.json"),
    )
    args = parser.parse_args()

    # Prefer the project key when this script is run from a shell that has an
    # older OPENAI_API_KEY already set.
    load_dotenv(REPOSITORY_ROOT / "config" / ".env", override=True)
    cfg = load_config(args.config)
    documents = load_parsed_documents(args.config)
    knowledge_cfg = cfg["knowledge"]
    chunks = chunk_documents(
        documents,
        chunk_size=int(knowledge_cfg.get("chunk_size", 800)),
        chunk_overlap=int(knowledge_cfg.get("chunk_overlap", 120)),
    )
    decisions = filter_documents(chunks)
    if not args.skip_descriptions:
        client = openai.OpenAI(
            api_key=require_env("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL", "https://us.api.openai.com/v1"),
        )
        decisions = add_llm_descriptions(
            decisions,
            client=client,
            model=args.model or cfg["model"]["llm_model"],
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    retained = [decision for decision in decisions if decision.retained]
    output = {
        "threshold": 0.5,
        "min_positive_labels": 1,
        "description_stage_enabled": not args.skip_descriptions,
        "description_model": None if args.skip_descriptions else (args.model or cfg["model"]["llm_model"]),
        "unit": "chunk",
        "source_documents": len(documents),
        "chunk_size": int(knowledge_cfg.get("chunk_size", 800)),
        "chunk_overlap": int(knowledge_cfg.get("chunk_overlap", 120)),
        "input_passages": len(decisions),
        "retained_passages": len(retained),
        "decisions": [asdict(decision) for decision in decisions],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Classified {len(decisions)} passages; retained {len(retained)}.")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
