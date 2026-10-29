"""Reproducible research primitives for the FLIPMM reliability study.

This module deliberately contains no LLM calls and no physical assumptions.  It
operates on explicit data, split, evidence, and feasibility contracts so that
the retrospective study can be audited before any model or literature result
is interpreted scientifically.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import ast
import copy
from itertools import combinations
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator
from scipy.spatial import ConvexHull, Delaunay
from sklearn.metrics import average_precision_score, brier_score_loss, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, r2_score


CANONICAL_INPUTS = ("water_speed", "pulse_energy", "frequency", "scanning_speed")
CANONICAL_OUTPUTS = ("width", "depth", "mrr", "haz")


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def corpus_manifest(root: Path) -> pd.DataFrame:
    """Return one row per PDF with hash, metadata, and available text status."""
    rows = []
    for path in sorted(root.glob("*.pdf")):
        text_available = False
        title = ""
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            text_available = any(bool(page.extract_text()) for page in reader.pages)
            title = str((reader.metadata or {}).get("/Title") or "")
        except Exception as exc:  # metadata audit must not hide unreadable files
            title = f"UNREADABLE: {type(exc).__name__}"
        rows.append(
            {
                "file": path.name,
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "title": title,
                "page_text_available": text_available,
                "process_tag": "uncertain",
                "source_role": "unresolved_lineage",
            }
        )
    return pd.DataFrame(rows)


def audit_table(frame: pd.DataFrame, input_columns: Sequence[str], output_columns: Sequence[str]) -> dict:
    """Create a conservative audit; values are summarized, never imputed."""
    required = list(input_columns) + list(output_columns)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing configured columns: {missing}")
    numeric = frame[required].apply(pd.to_numeric, errors="coerce")
    summary = {
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "missing_columns": missing,
        "missing_values": numeric.isna().sum().to_dict(),
        "duplicate_rows": int(frame.duplicated().sum()),
        "constant_columns": [c for c in required if numeric[c].nunique(dropna=True) <= 1],
        "negative_values": {c: int((numeric[c] < 0).sum()) for c in required},
        "zero_values": {c: int((numeric[c] == 0).sum()) for c in required},
        "unique_levels": {c: int(numeric[c].nunique(dropna=True)) for c in input_columns},
        "descriptive_statistics": numeric.describe().to_dict(),
        "correlation": numeric.corr().to_dict(),
    }
    summary["mrr_recomputable_from_geometry"] = all(
        c in frame.columns for c in ("Width (micrometer)", "Depth (micrometer)", "ScanningSpeed (mm/s)")
    )
    return summary


@dataclass(frozen=True)
class AxisSplit:
    axis: str
    boundary: float
    retained_span: float
    test_levels: tuple[float, ...]
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    test_indices: tuple[int, ...]


def single_axis_split(
    frame: pd.DataFrame,
    axis: str,
    group_column: str | None = None,
    holdout_levels: int = 1,
    validation_fraction: float = 0.25,
    random_state: int = 50,
) -> AxisSplit:
    """Hold out complete highest axis levels, keeping groups intact.

    Rows sharing a level or supplied group are never divided across partitions.
    The returned indices reference the original frame.
    """
    if holdout_levels < 1 or not 0 < validation_fraction < 1:
        raise ValueError("holdout_levels must be positive and validation_fraction in (0, 1)")
    values = pd.to_numeric(frame[axis], errors="raise")
    levels = np.sort(values.dropna().unique())
    if len(levels) <= holdout_levels + 1:
        raise ValueError("Need at least two retained levels and the requested held-out levels")
    test_levels = tuple(float(x) for x in levels[-holdout_levels:])
    retained_levels = levels[:-holdout_levels]
    test_mask = values.isin(test_levels)
    retained = frame.index[~test_mask].to_numpy()
    if group_column is None:
        groups = pd.Series(frame.index, index=frame.index)
    else:
        if group_column not in frame:
            raise ValueError(f"Unknown group column: {group_column}")
        groups = frame[group_column].astype("string")
    # A group touching the test boundary belongs wholly to test.
    test_groups = set(groups.loc[test_mask].tolist())
    test_mask |= groups.isin(test_groups)
    retained = frame.index[~test_mask].to_numpy()
    rng = np.random.default_rng(random_state)
    unique_groups = np.array(sorted(set(groups.loc[retained].tolist()), key=str), dtype=object)
    rng.shuffle(unique_groups)
    n_validation = max(1, int(round(len(unique_groups) * validation_fraction)))
    val_groups = set(unique_groups[:n_validation].tolist())
    validation_mask = groups.isin(val_groups) & ~test_mask
    train_mask = ~test_mask & ~validation_mask
    train_levels = values.loc[train_mask].unique()
    if len(train_levels) < 2:
        raise ValueError("Split leaves fewer than two training levels")
    retained_span = float(np.max(values.loc[train_mask]) - np.min(values.loc[train_mask]))
    return AxisSplit(
        axis=axis,
        boundary=float(np.min(values.loc[test_mask])),
        retained_span=retained_span,
        test_levels=test_levels,
        train_indices=tuple(int(i) for i in frame.index[train_mask]),
        validation_indices=tuple(int(i) for i in frame.index[validation_mask]),
        test_indices=tuple(int(i) for i in frame.index[test_mask]),
    )


def scaled_nearest_distance(train_x: np.ndarray, query_x: np.ndarray) -> np.ndarray:
    """Nearest-neighbor distance after scaling on training data only."""
    scaler = StandardScaler().fit(train_x)
    train_z, query_z = scaler.transform(train_x), scaler.transform(query_x)
    return np.min(np.linalg.norm(query_z[:, None, :] - train_z[None, :, :], axis=2), axis=1)


@dataclass
class RBFTeacher:
    model: RBFInterpolator
    x_scaler: StandardScaler
    y_mean: float
    y_scale: float
    hull: Delaunay
    bounds: np.ndarray

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.model(self.x_scaler.transform(np.asarray(x))) * self.y_scale + self.y_mean

    def contains(self, x: np.ndarray) -> np.ndarray:
        return self.hull.find_simplex(self.x_scaler.transform(np.asarray(x))) >= 0


def fit_rbf_teacher(x: np.ndarray, y: np.ndarray, kernel: str = "thin_plate_spline", smoothing: float = 0.1) -> RBFTeacher:
    """Fit an explicitly marked teacher on a designated reference subset."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < x.shape[1] + 1:
        raise ValueError("RBF teacher requires more points than input dimensions")
    x_scaler = StandardScaler().fit(x)
    y_mean, y_scale = float(np.mean(y)), float(np.std(y) or 1.0)
    x_z = x_scaler.transform(x)
    model = RBFInterpolator(x_z, (y - y_mean) / y_scale, kernel=kernel, smoothing=smoothing)
    hull = Delaunay(x_z)
    return RBFTeacher(model, x_scaler, y_mean, y_scale, hull, np.column_stack([x.min(0), x.max(0)]))


def latin_hypercube_in_hull(x: np.ndarray, n: int, rng: np.random.Generator, feasible: Callable[[np.ndarray], np.ndarray] | None = None) -> np.ndarray:
    """Generate synthetic development candidates inside the observed hull."""
    x = np.asarray(x, dtype=float)
    scaler = StandardScaler().fit(x)
    z = scaler.transform(x)
    hull = Delaunay(z)
    result: list[np.ndarray] = []
    attempts = 0
    while len(result) < n and attempts < max(1000, n * 100):
        attempts += 1
        unit = (rng.random(x.shape[1]) + rng.permutation(n))[: x.shape[1]] / max(n, 1)
        candidate = x.min(0) + unit * (x.max(0) - x.min(0))
        if hull.find_simplex(scaler.transform(candidate[None, :]))[0] < 0:
            continue
        if feasible is not None and not bool(np.asarray(feasible(candidate[None, :])).ravel()[0]):
            continue
        result.append(candidate)
    if len(result) < n:
        raise RuntimeError("Could not generate requested feasible convex-hull candidates")
    return np.vstack(result)


@dataclass(frozen=True)
class Claim:
    claim_id: str
    source: str
    page: int | None
    quote: str
    parameter: str | None
    target: str | None
    relationship: str | None
    observed_or_inferred: str
    applicability: str
    lineage_role: str


def validate_claims(claims: Iterable[Claim], source_text: dict[str, str], allowed_parameters: set[str], allowed_targets: set[str]) -> list[str]:
    """Return validation errors; exact quote and provenance are mandatory."""
    errors: list[str] = []
    seen: set[str] = set()
    for claim in claims:
        if claim.claim_id in seen:
            errors.append(f"duplicate claim_id: {claim.claim_id}")
        seen.add(claim.claim_id)
        text = source_text.get(claim.source, "")
        if not text:
            errors.append(f"missing source text: {claim.claim_id}")
        elif claim.quote not in text:
            errors.append(f"quote not found in source: {claim.claim_id}")
        if claim.parameter is not None and claim.parameter not in allowed_parameters:
            errors.append(f"invalid parameter: {claim.claim_id}")
        if claim.target is not None and claim.target not in allowed_targets:
            errors.append(f"invalid target: {claim.claim_id}")
        if claim.applicability not in {"supported", "unresolved", "challenged"}:
            errors.append(f"invalid applicability: {claim.claim_id}")
    return errors


def committee_disagreement(predictions: np.ndarray, tau: float, epsilon: float = 1e-12) -> pd.DataFrame:
    """Compute fixed-predictor committee disagreement without test outcomes."""
    p = np.asarray(predictions, dtype=float)
    if p.ndim != 2:
        raise ValueError("predictions must have shape (models, points)")
    mean = p.mean(0)
    spread = p.std(0)
    result = pd.DataFrame({"mean": mean, "absolute_spread": spread, "tolerance_scaled_spread": spread / tau})
    result["paper_cv"] = np.where((mean > epsilon), spread / mean, np.nan)
    pair_values = []
    for column in p.T:
        vals = [math.log1p(abs(a - b) / max(a, b)) if min(a, b) > 0 else np.nan for i, a in enumerate(column) for b in column[i + 1 :]]
        pair_values.append(np.nanmean(vals) if vals and not np.all(np.isnan(vals)) else np.nan)
    result["paper_ibmd"] = pair_values
    result["robust_iqr_scaled"] = (np.percentile(p, 75, axis=0) - np.percentile(p, 25, axis=0)) / tau
    result["robust_mad_scaled"] = np.median(np.abs(p - np.median(p, axis=0)), axis=0) / tau
    return result


def variance_decomposition(means: np.ndarray, within_variances: np.ndarray, weights: np.ndarray | None = None) -> pd.DataFrame:
    """Conditional within/between decomposition for a declared finite mixture."""
    means, within_variances = np.asarray(means, float), np.asarray(within_variances, float)
    if means.shape != within_variances.shape or means.ndim != 2:
        raise ValueError("means and within_variances must both be (models, points)")
    w = np.ones(means.shape[0]) / means.shape[0] if weights is None else np.asarray(weights, float)
    w = w / w.sum()
    mixture_mean = np.sum(w[:, None] * means, axis=0)
    within = np.sum(w[:, None] * within_variances, axis=0)
    between = np.sum(w[:, None] * (means - mixture_mean) ** 2, axis=0)
    return pd.DataFrame({"within": within, "between": between, "total": within + between})


def rank_metrics(score: Sequence[float], absolute_error: Sequence[float], tau: float) -> dict:
    """Outcome evaluation for a frozen score, including prevalence and coverage."""
    score, error = np.asarray(score, float), np.asarray(absolute_error, float)
    failed = error > tau
    result = {"failure_prevalence": float(failed.mean()), "n": int(len(error))}
    result["average_precision"] = float(average_precision_score(failed, score)) if failed.any() and (~failed).any() else None
    order = np.argsort(-score)
    result["risk_coverage"] = [
        {"coverage": float(k / len(error)), "failure_rate": float(failed[order[:k]].mean())}
        for k in range(1, len(error) + 1)
    ]
    return result


def development_threshold_grid(y_development: Sequence[float], fractions: Sequence[float] = (0.1, 0.25, 0.5, 1.0)) -> pd.DataFrame:
    """Create statistical absolute-error thresholds from development outcomes only."""
    y = np.asarray(y_development, float)
    scale = float(np.nanstd(y, ddof=1))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Development output scale must be finite and positive")
    return pd.DataFrame({"scale_fraction": list(fractions), "tau": [float(scale * fraction) for fraction in fractions], "interpretation": "statistical_threshold_not_engineering_tolerance"})


def acquisition_scores(predictions: np.ndarray, tau: float, strategy: str, observed: np.ndarray | None = None) -> np.ndarray:
    """Label-blind acquisition scores for random, space-filling, and QBC policies."""
    p = np.asarray(predictions, float)
    if strategy == "random":
        return np.random.default_rng(0).random(p.shape[1])
    if strategy in {"qbc_std", "tolerance_scaled", "robust"}:
        metrics = committee_disagreement(p, tau)
        column = {"qbc_std": "absolute_spread", "tolerance_scaled": "tolerance_scaled_spread", "robust": "robust_iqr_scaled"}[strategy]
        return metrics[column].fillna(-np.inf).to_numpy()
    if strategy == "space_filling":
        if observed is None or len(observed) == 0:
            return np.ones(p.shape[1])
        return scaled_nearest_distance(observed, p.T)
    raise ValueError(f"Unknown acquisition strategy: {strategy}")


# ---------------------------------------------------------------------------
# Study-contract additions
# ---------------------------------------------------------------------------

def make_data_dictionary(frame: pd.DataFrame, input_columns: Sequence[str], output_columns: Sequence[str]) -> dict:
    """Return an auditable column dictionary without inventing unavailable metadata."""
    required = list(input_columns) + list(output_columns)
    return {
        "columns": {
            column: {
                "present": column in frame.columns,
                "dtype": str(frame[column].dtype) if column in frame else None,
                "unit_from_header": column[column.find("(") + 1 : column.rfind(")")] if "(" in column and ")" in column else None,
                "role": "input" if column in input_columns else "output",
                "metadata_status": "unreported",
            }
            for column in required
        },
        "unavailable_metadata": [
            "material", "pulse_duration", "wavelength", "spot_size", "focusing_geometry",
            "water_conditions", "passes", "instrument", "measurement_procedure", "specimen_id", "run_id",
        ],
        "response_lineage": {column: "unclassified" for column in output_columns},
    }


def task_manifests(
    frame: pd.DataFrame,
    input_columns: Sequence[str],
    output_columns: Sequence[str],
    group_column: str | None = None,
    holdout_levels: int = 1,
    validation_fraction: float = 0.25,
    seed: int = 50,
) -> pd.DataFrame:
    """Create one row per output/axis task with split hashes and distances."""
    rows: list[dict] = []
    for output in output_columns:
        for axis in input_columns:
            split = single_axis_split(frame, axis, group_column, holdout_levels, validation_fraction, seed)
            train = frame.loc[list(split.train_indices), list(input_columns)]
            test = frame.loc[list(split.test_indices), list(input_columns)]
            axis_values = pd.to_numeric(frame[axis], errors="raise")
            rows.append({
                "task_id": f"{output}__{axis}__seed{seed}",
                "output": output,
                "axis": axis,
                "seed": seed,
                "train_n": len(split.train_indices),
                "validation_n": len(split.validation_indices),
                "test_n": len(split.test_indices),
                "boundary": split.boundary,
                "retained_span": split.retained_span,
                "test_distance_over_span": float((axis_values.loc[list(split.test_indices)].min() - axis_values.loc[list(split.train_indices)].max()) / max(split.retained_span, np.finfo(float).eps)),
                "nearest_distance_mean": float(scaled_nearest_distance(train.to_numpy(float), test.to_numpy(float)).mean()),
                "other_axis_violations": int(sum(
                    int((test[column].min() < train[column].min()) or (test[column].max() > train[column].max()))
                    for column in input_columns if column != axis
                )),
                "train_indices_sha256": hashlib.sha256(json.dumps(split.train_indices).encode()).hexdigest(),
                "validation_indices_sha256": hashlib.sha256(json.dumps(split.validation_indices).encode()).hexdigest(),
                "test_indices_sha256": hashlib.sha256(json.dumps(split.test_indices).encode()).hexdigest(),
            })
    return pd.DataFrame(rows)


def grouped_folds(indices: Sequence[int], groups: Sequence[object], n_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return grouped folds, reducing the fold count when the data are small."""
    unique = len(set(groups))
    n_splits = min(n_splits, unique)
    if n_splits < 2:
        return []
    splitter = GroupKFold(n_splits=n_splits)
    x = np.zeros((len(indices), 1))
    return [(np.asarray(indices)[train], np.asarray(indices)[test]) for train, test in splitter.split(x, groups=groups)]


def select_rbf_hyperparameters(
    x: np.ndarray,
    y: np.ndarray,
    groups: Sequence[object] | None = None,
    kernels: Sequence[str] = ("linear", "thin_plate_spline", "cubic"),
    smoothings: Sequence[float] = (0.001, 0.1, 1.0),
    n_splits: int = 5,
) -> pd.DataFrame:
    """Grouped-CV selection for the development-only RBF arm."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    groups = np.arange(len(x)) if groups is None else np.asarray(groups)
    folds = grouped_folds(np.arange(len(x)), groups, n_splits)
    if not folds:
        raise ValueError("RBF CV requires at least two independent groups")
    rows = []
    for kernel in kernels:
        for smoothing in smoothings:
            errors = []
            for train, validation in folds:
                scaler = StandardScaler().fit(x[train])
                y_mean, y_scale = y[train].mean(), y[train].std() or 1.0
                model = RBFInterpolator(scaler.transform(x[train]), (y[train] - y_mean) / y_scale, kernel=kernel, smoothing=smoothing)
                predicted = model(scaler.transform(x[validation])) * y_scale + y_mean
                errors.append(mean_squared_error(y[validation], predicted))
            rows.append({"kernel": kernel, "smoothing": smoothing, "cv_mse": float(np.mean(errors)), "folds": len(errors)})
    return pd.DataFrame(rows).sort_values("cv_mse", ignore_index=True)


def interval_metrics(y_true: Sequence[float], lower: Sequence[float], upper: Sequence[float], nominal: float) -> dict:
    """Evaluate interval coverage, width, and interval score."""
    y, lo, hi = map(lambda value: np.asarray(value, float), (y_true, lower, upper))
    if np.any(lo > hi) or not 0 < nominal < 1:
        raise ValueError("Require ordered bounds and nominal level in (0, 1)")
    alpha = 1 - nominal
    score = (hi - lo) + (2 / alpha) * (lo - y) * (y < lo) + (2 / alpha) * (y - hi) * (y > hi)
    return {
        "nominal": nominal,
        "coverage": float(np.mean((y >= lo) & (y <= hi))),
        "mean_width": float(np.mean(hi - lo)),
        "median_width": float(np.median(hi - lo)),
        "interval_score": float(np.mean(score)),
    }


def evidence_assessment(claims: Iterable[Claim], candidate_assumptions: Sequence[str]) -> dict:
    """Map verified claims to an ordinal diagnostic; never produce probabilities."""
    relevant = [claim for claim in claims if claim.parameter in candidate_assumptions or claim.target in candidate_assumptions]
    levels = {claim.applicability for claim in relevant}
    if "challenged" in levels:
        level = "challenged"
    elif "supported" in levels:
        level = "supported"
    else:
        level = "unresolved"
    return {"level": level, "claim_ids": [claim.claim_id for claim in relevant], "justification_required": not bool(relevant)}


def candidate_metadata(code: str, *, candidate_id: str, source: str, seed: int | None, units: dict, assumptions: Sequence[str], validation_score: float | None = None) -> dict:
    """Build a complete candidate archive record without executing code."""
    return {
        "candidate_id": candidate_id,
        "equation": code,
        "source": source,
        "generation_seed": seed,
        "coefficient_definitions": {},
        "bounds": {},
        "units": units,
        "assumptions": list(assumptions),
        "validation_score": validation_score,
        "fit_status": "unreviewed",
        "retrieval_sources": [],
    }


_SAFE_NODES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.USub, ast.UAdd, ast.Constant, ast.Name, ast.Load, ast.Call, ast.Tuple)
_SAFE_FUNCTIONS = {"abs", "exp", "log", "sqrt", "sin", "cos", "tan", "maximum", "minimum"}


def validate_expression(expression: str, allowed_names: Iterable[str]) -> None:
    """Reject imports, attributes, statements, and calls outside the safe subset."""
    tree = ast.parse(expression, mode="eval")
    names = set(allowed_names)
    for node in ast.walk(tree):
        if not isinstance(node, _SAFE_NODES):
            raise ValueError(f"Unsupported expression node: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in names and node.id not in _SAFE_FUNCTIONS:
            raise ValueError(f"Unknown expression name: {node.id}")
        if isinstance(node, ast.Call) and not isinstance(node.func, ast.Name):
            raise ValueError("Only direct calls to approved functions are allowed")
        if isinstance(node, ast.Call) and node.func.id not in _SAFE_FUNCTIONS:
            raise ValueError(f"Unapproved expression function: {node.func.id}")


def constraint_violation(prediction: np.ndarray, *, nonnegative: bool = False, lower: float | None = None, upper: float | None = None) -> np.ndarray:
    """Return normalized prediction violations; constraints are opt-in."""
    prediction = np.asarray(prediction, float)
    violations = np.zeros_like(prediction)
    if nonnegative:
        violations += np.maximum(-prediction, 0)
    if lower is not None:
        violations += np.maximum(lower - prediction, 0)
    if upper is not None:
        violations += np.maximum(prediction - upper, 0)
    scale = max(float(np.nanstd(prediction)), np.finfo(float).eps)
    return violations / scale


def pareto_front(scores: Sequence[float], complexities: Sequence[float]) -> np.ndarray:
    """Return accuracy-complexity Pareto membership, minimizing both quantities."""
    scores, complexities = np.asarray(scores, float), np.asarray(complexities, float)
    if scores.shape != complexities.shape:
        raise ValueError("scores and complexities must have the same shape")
    keep = np.ones(len(scores), dtype=bool)
    for i in range(len(scores)):
        dominated = (scores <= scores[i]) & (complexities <= complexities[i]) & ((scores < scores[i]) | (complexities < complexities[i]))
        keep[i] = not np.any(dominated)
    return keep


def pool_based_replay(
    x_initial: np.ndarray,
    y_initial: np.ndarray,
    x_pool: np.ndarray,
    y_pool: np.ndarray,
    x_locked: np.ndarray,
    y_locked: np.ndarray,
    fit_predict: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    acquisition: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    budget: int,
) -> pd.DataFrame:
    """Replay pool-based active learning with labels hidden from acquisition.

    ``fit_predict`` receives training data and returns committee predictions on
    arbitrary points. ``acquisition`` receives committee predictions on the
    remaining pool and returns one score per pool point. Locked outcomes are
    used only for the reported endpoint after each query.
    """
    x_train, y_train = np.asarray(x_initial).copy(), np.asarray(y_initial).copy()
    x_remaining, y_remaining = np.asarray(x_pool).copy(), np.asarray(y_pool).copy()
    rows = []
    for step in range(min(budget, len(x_remaining)) + 1):
        locked_predictions = np.asarray(fit_predict(x_train, y_train, x_locked))
        locked_mean = locked_predictions.mean(0) if locked_predictions.ndim == 2 else locked_predictions
        rows.append({"step": step, "n_observed": len(x_train), "mae": float(mean_absolute_error(y_locked, locked_mean)), "failure_rate": float(np.mean(np.abs(y_locked - locked_mean) > np.std(y_train)))})
        if step == budget or len(x_remaining) == 0:
            break
        pool_predictions = np.asarray(fit_predict(x_train, y_train, x_remaining))
        scores = np.asarray(acquisition(pool_predictions, x_train, x_remaining), float)
        choice = int(np.nanargmax(scores))
        x_train = np.vstack([x_train, x_remaining[choice]])
        y_train = np.append(y_train, y_remaining[choice])
        x_remaining = np.delete(x_remaining, choice, axis=0)
        y_remaining = np.delete(y_remaining, choice, axis=0)
    return pd.DataFrame(rows)
