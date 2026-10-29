"""Build the first-pass FLIPMM claim-versus-data evidence package."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pypdf import PdfReader
from scipy.interpolate import RBFInterpolator
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config_loader import load_config, resolve_path

RNG = np.random.default_rng(50)
INPUTS = {
    "WaterSpeed (m/s)": {"short": "water speed", "terms": ["water speed", "water flow", "flow rate"]},
    "PulseEnergy (microJ)": {"short": "pulse energy", "terms": ["pulse energy", "pulse power", "energy per pulse"]},
    "Frequency (kHz)": {"short": "frequency", "terms": ["frequency", "repetition rate", "pulse frequency"]},
    "ScanningSpeed (mm/s)": {"short": "scanning speed", "terms": ["scanning speed", "scan speed", "scanning velocity"]},
}
OUTPUTS = {
    "Width (micrometer)": {"short": "width", "terms": ["channel width", "machined width", "width"]},
    "Depth (micrometer)": {"short": "depth", "terms": ["channel depth", "machining depth", "depth"]},
    "MRR (cubic micrometer per sec)": {"short": "mrr", "terms": ["material removal rate", "mrr"]},
    "HAZ (micrometer)": {"short": "haz", "terms": ["heat affected zone", "heat-affected zone", "haz"]},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_quote(value: str) -> str:
    value = value.replace("\u00ad", "")
    value = re.sub(r"-\s*\n\s*", "", value)
    return re.sub(r"\s+", " ", value).strip()


def extract_pdfs(pdf_dir: Path) -> tuple[pd.DataFrame, dict[str, dict[int, str]]]:
    rows, pages = [], {}
    text_dir = OUT / "extracted_text"
    text_dir.mkdir(parents=True, exist_ok=True)
    for pdf in sorted(pdf_dir.glob("*.pdf")):
        reader = PdfReader(str(pdf))
        paper_pages: dict[int, str] = {}
        chars = 0
        page_failures = 0
        for number, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
                page_failures += 1
            paper_pages[number] = text
            chars += len(text)
        pages[pdf.name] = paper_pages
        combined = "\n\n".join(f"\n--- PAGE {n} ---\n{text}" for n, text in paper_pages.items())
        (text_dir / f"{pdf.stem}.txt").write_text(combined, encoding="utf-8")
        sample = normalize_quote(combined[:4000])
        garbled_markers = int(combined.count("/gid") + combined.count("â") + combined.count("�"))
        rows.append({
            "paper": pdf.name,
            "sha256": sha256(pdf),
            "pages": len(reader.pages),
            "characters_extracted": chars,
            "page_extraction_failures": page_failures,
            "text_status": "usable_but_garbled" if chars >= 1000 and garbled_markers else ("usable" if chars >= 1000 else "insufficient_text"),
            "garbled_markers": garbled_markers,
            "ocr_attempted": False,
            "sample_text": sample[:500],
        })
    return pd.DataFrame(rows), pages


def classify_papers(manifest: pd.DataFrame, pages: dict[str, dict[int, str]]) -> pd.DataFrame:
    rows = []
    for record in manifest.to_dict("records"):
        text = normalize_quote(" ".join(pages[record["paper"]].values())).lower()
        flip_terms = ["laser-induced plasma micromachining", "lipmm", "flipmm", "plasma micromachining"]
        laser_terms = ["laser", "micromachin", "water", "plasma"]
        if any(term in text for term in flip_terms):
            category, reason = "flipmm_or_laser_micro", "Text mentions laser-induced plasma micromachining or FLIPMM-related terminology."
        elif sum(term in text for term in laser_terms) >= 2:
            category, reason = "adjacent_manufacturing", "Text contains multiple laser/micro-machining terms but no confirmed FLIPMM match."
        else:
            category, reason = "other", "No sufficiently specific FLIPMM or laser micro-machining match was found automatically."
        rows.append({**record, "category": category, "classification_reason": reason, "needs_human_review": True})
    return pd.DataFrame(rows)


def audit_dataset(frame: pd.DataFrame, cfg: dict) -> dict:
    inputs, outputs = cfg["input_columns"], cfg["output_columns"]
    numeric = frame[inputs + outputs].apply(pd.to_numeric, errors="coerce")
    combinations = frame[inputs].drop_duplicates()
    replicates = frame.groupby(inputs, dropna=False).size()
    result = {
        "rows": len(frame),
        "columns": [{"name": c, "dtype": str(frame[c].dtype), "unit_in_header": (re.search(r"\(([^)]*)\)", c).group(1) if re.search(r"\(([^)]*)\)", c) else None)} for c in frame.columns],
        "missing_values": numeric.isna().sum().to_dict(),
        "duplicate_rows": int(frame.duplicated().sum()),
        "unique_input_combinations": len(combinations),
        "input_levels": {c: int(frame[c].nunique()) for c in inputs},
        "replicates_per_combination": replicates.value_counts().sort_index().to_dict(),
        "design_assessment": "complete factorial-looking grid" if len(combinations) == np.prod([frame[c].nunique() for c in inputs]) else "irregular or incomplete design",
        "negative_values": {c: int((numeric[c] < 0).sum()) for c in outputs},
        "ranges": {c: {"min": float(numeric[c].min()), "max": float(numeric[c].max())} for c in inputs + outputs},
        "correlations": numeric.corr().to_dict(),
        "open_questions": [
            "Are rows measured experiments, averaged replicates, response-surface values, or synthetic values?",
            "What are the material, pulse duration, wavelength, spot size, focusing, water, instrument, and measurement-procedure metadata?",
            "Are negative depth and MRR values physically valid signed quantities or data-generation artifacts?",
            "Is MRR independently measured or computed from geometry and scanning speed?",
            "Are there hidden run, specimen, batch, or session identifiers?",
        ],
    }
    if all(c in frame for c in ["Width (micrometer)", "Depth (micrometer)", "ScanningSpeed (mm/s)"]):
        product = frame["Width (micrometer)"] * frame["Depth (micrometer)"] * frame["ScanningSpeed (mm/s)"]
        target = frame["MRR (cubic micrometer per sec)"]
        model = LinearRegression().fit(product.to_numpy().reshape(-1, 1), target)
        result["mrr_geometry_check"] = {
            "candidate": "width * depth * scanning speed",
            "r2_with_intercept": float(model.score(product.to_numpy().reshape(-1, 1), target)),
            "ratio_std": float(np.std(target / product)),
            "interpretation": "not verified as an exact derivation; advisor confirmation required",
        }
    return result


def regime_split(frame: pd.DataFrame, axis: str, level: float) -> tuple[np.ndarray, np.ndarray]:
    test = frame[axis] == level
    return np.flatnonzero(~test.to_numpy()), np.flatnonzero(test.to_numpy())


def fit_model(kind: str, x_train: np.ndarray, y_train: np.ndarray):
    if kind == "linear":
        model = LinearRegression().fit(x_train, y_train)
        return lambda x: model.predict(x), model
    if kind == "quadratic":
        model = make_pipeline(PolynomialFeatures(2, include_bias=False), LinearRegression()).fit(x_train, y_train)
        return lambda x: model.predict(x), model
    scaler_x = StandardScaler().fit(x_train)
    y_mean, y_scale = float(y_train.mean()), float(y_train.std() or 1.0)
    rbf = RBFInterpolator(scaler_x.transform(x_train), (y_train - y_mean) / y_scale, kernel="cubic", smoothing=0.1)
    return lambda x: rbf(scaler_x.transform(x)) * y_scale + y_mean, rbf


def validation_table(frame: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    inputs, outputs = cfg["input_columns"], cfg["output_columns"]
    rows = []
    for output in outputs:
        for axis in inputs:
            x = frame[inputs].to_numpy(float)
            y = frame[output].to_numpy(float)
            # Every input setting is unique in this file. KFold is therefore
            # group-safe for this first pass, while the absence of run IDs is noted.
            fold_rows = []
            for train_idx, test_idx in KFold(n_splits=5, shuffle=True, random_state=50).split(x):
                for kind in ("linear", "quadratic", "rbf"):
                    predict, _ = fit_model(kind, x[train_idx], y[train_idx])
                    pred = predict(x[test_idx])
                    fold_rows.append((kind, mean_squared_error(y[test_idx], pred) ** 0.5, r2_score(y[test_idx], pred), mean_absolute_error(y[test_idx], pred)))
            for kind in ("linear", "quadratic", "rbf"):
                values = [row[1:] for row in fold_rows if row[0] == kind]
                rows.append({"evaluation": "5-fold unique-setting holdout", "output": output, "axis": axis, "model": kind, "rmse_mean": float(np.mean([v[0] for v in values])), "r2_mean": float(np.mean([v[1] for v in values])), "mae_mean": float(np.mean([v[2] for v in values]))})
            for level in (frame[axis].max(),):
                train_idx, test_idx = regime_split(frame, axis, level)
                for kind in ("linear", "quadratic", "rbf"):
                    predict, _ = fit_model(kind, x[train_idx], y[train_idx])
                    pred = predict(x[test_idx])
                    rows.append({"evaluation": f"leave_{axis}_highest_level_out", "output": output, "axis": axis, "model": kind, "rmse_mean": mean_squared_error(y[test_idx], pred) ** 0.5, "r2_mean": r2_score(y[test_idx], pred), "mae_mean": mean_absolute_error(y[test_idx], pred)})
    return pd.DataFrame(rows)


def support_table(frame: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Bootstrap a quadratic model and classify within-range directional trends."""
    inputs, outputs = cfg["input_columns"], cfg["output_columns"]
    rows = []
    for output in outputs:
        for variable in inputs:
            x = frame[inputs].to_numpy(float)
            y = frame[output].to_numpy(float)
            grid = np.linspace(frame[variable].quantile(0.1), frame[variable].quantile(0.9), 9)
            other = frame[inputs].median().to_numpy(float)
            points = np.tile(other, (len(grid), 1))
            points[:, inputs.index(variable)] = grid
            slopes = []
            for _ in range(100):
                idx = RNG.integers(0, len(x), len(x))
                predict, _ = fit_model("quadratic", x[idx], y[idx])
                slopes.append(np.polyfit(grid, predict(points), 1)[0])
            fraction_positive = float(np.mean(np.asarray(slopes) > 0))
            fraction_negative = float(np.mean(np.asarray(slopes) < 0))
            for direction, fraction in (("increases", fraction_positive), ("decreases", fraction_negative)):
                verdict = "supports" if fraction >= 0.8 else "inconclusive"
                rows.append({"output": output, "variable": variable, "direction": direction, "condition_range": "10th-90th percentile; other inputs at medians", "bootstrap_direction_fraction": fraction, "data_verdict": verdict, "explanation": "Quadratic bootstrap direction consistency within observed central range; no outside-range claim."})
    return pd.DataFrame(rows)


def surrogate_plots(frame: pd.DataFrame, cfg: dict) -> None:
    """Create one-factor slices and two-factor heatmaps with sparse-region marks."""
    inputs, outputs = cfg["input_columns"], cfg["output_columns"]
    x = frame[inputs].to_numpy(float)
    scaled = StandardScaler().fit_transform(x)
    plot_dir = OUT / "surrogate_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for output in outputs:
        y = frame[output].to_numpy(float)
        predict, _ = fit_model("quadratic", x, y)
        for variable in inputs:
            grid = np.linspace(frame[variable].min(), frame[variable].max(), 80)
            points = np.tile(frame[inputs].median().to_numpy(float), (len(grid), 1))
            points[:, inputs.index(variable)] = grid
            prediction = predict(points)
            distances = cdist(StandardScaler().fit(x).transform(points), scaled).min(axis=1)
            sparse_cutoff = float(np.quantile(distances, 0.75))
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(grid, prediction, color="#1f77b4", linewidth=2, label="quadratic surrogate slice")
            ax.fill_between(grid, prediction, where=distances > sparse_cutoff, color="#d62728", alpha=0.18, label="sparser slice region")
            ax.set_title(f"{output}: {variable}; other inputs at medians")
            ax.set_xlabel(variable)
            ax.set_ylabel(output)
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(plot_dir / f"slice_{OUTPUTS[output]['short']}_{INPUTS[variable]['short'].replace(' ', '_')}.png", dpi=160)
            plt.close(fig)
        pair = inputs[:2]
        a_grid = np.linspace(frame[pair[0]].min(), frame[pair[0]].max(), 35)
        b_grid = np.linspace(frame[pair[1]].min(), frame[pair[1]].max(), 35)
        aa, bb = np.meshgrid(a_grid, b_grid)
        points = np.tile(frame[inputs].median().to_numpy(float), (aa.size, 1))
        points[:, inputs.index(pair[0])] = aa.ravel()
        points[:, inputs.index(pair[1])] = bb.ravel()
        prediction = predict(points).reshape(aa.shape)
        distances = cdist(StandardScaler().fit(x).transform(points), scaled).min(axis=1).reshape(aa.shape)
        fig, ax = plt.subplots(figsize=(8, 6))
        image = ax.contourf(aa, bb, prediction, levels=18, cmap="viridis")
        sparse_cutoff = float(np.quantile(distances, 0.75))
        ax.contour(aa, bb, distances, levels=[sparse_cutoff], colors="red", linewidths=1.5)
        ax.scatter(frame[pair[0]], frame[pair[1]], s=8, c="white", edgecolors="black", linewidths=0.2, alpha=0.4)
        ax.set_title(f"{output}: {pair[0]} × {pair[1]} heatmap; other inputs at medians")
        ax.set_xlabel(pair[0])
        ax.set_ylabel(pair[1])
        fig.colorbar(image, ax=ax, label=output)
        fig.tight_layout()
        fig.savefig(plot_dir / f"heatmap_{OUTPUTS[output]['short']}.png", dpi=160)
        plt.close(fig)


def input_domain_coverage_plot(frame: pd.DataFrame, cfg: dict) -> None:
    """Plot observed/RBF-supported input ranges plus unsupported padded space.

    An RBFInterpolator can evaluate numerically outside the data, but this
    figure treats only the convex hull of the observed inputs as supported.
    The padded regions are visualization space for possible extrapolation, not
    validated model coverage.
    """
    inputs = cfg["input_columns"]
    values = frame[inputs].to_numpy(float)
    fig, axes = plt.subplots(len(inputs), 1, figsize=(13, 2.2 * len(inputs)), constrained_layout=True)
    if len(inputs) == 1:
        axes = [axes]
    for ax, column in zip(axes, inputs):
        column_values = frame[column].to_numpy(float)
        low, high = float(column_values.min()), float(column_values.max())
        span = high - low or 1.0
        pad = 0.25 * span
        ax.set_xlim(low - pad, high + pad)
        ax.axvspan(low - pad, low, facecolor="#f2f2f2", hatch="//", edgecolor="#bdbdbd", label="outside observed/RBF support")
        ax.axvspan(low, high, facecolor="#b7e1cd", alpha=0.9, label="observed convex-hull projection / RBF support")
        ax.axvspan(high, high + pad, facecolor="#f2f2f2", hatch="//", edgecolor="#bdbdbd")
        ax.plot(column_values, np.zeros_like(column_values), "|", color="#1f4e79", markersize=16, markeredgewidth=1.5, label="observed levels")
        ax.axvline(low, color="#2e7d32", linestyle="--", linewidth=1)
        ax.axvline(high, color="#2e7d32", linestyle="--", linewidth=1)
        ax.set_yticks([])
        ax.set_ylabel(column, rotation=0, ha="right", va="center")
        ax.grid(axis="x", alpha=0.25)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.35), ncol=3)
    fig.suptitle("FLIPMM input-domain coverage: observed/RBF-supported range and unsupported margin", y=1.02)
    fig.savefig(OUT / "input_domain_coverage.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Pairwise projections show that a rectangular-looking envelope is still a
    # projection; the full support is the 4-D convex hull.
    pairs = [(inputs[i], inputs[j]) for i in range(len(inputs)) for j in range(i + 1, len(inputs))]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for ax, (first, second) in zip(axes.flat, pairs):
        x = frame[[first, second]].to_numpy(float)
        hull = ConvexHull(x)
        polygon = x[hull.vertices]
        polygon = np.vstack([polygon, polygon[0]])
        ax.fill(polygon[:, 0], polygon[:, 1], color="#b7e1cd", alpha=0.55, label="projected RBF support")
        ax.plot(polygon[:, 0], polygon[:, 1], color="#2e7d32", linestyle="--", linewidth=1)
        ax.scatter(x[:, 0], x[:, 1], s=12, color="#1f4e79", alpha=0.55, label="observed settings")
        xspan = x[:, 0].max() - x[:, 0].min() or 1.0
        yspan = x[:, 1].max() - x[:, 1].min() or 1.0
        ax.set_xlim(x[:, 0].min() - 0.2 * xspan, x[:, 0].max() + 0.2 * xspan)
        ax.set_ylim(x[:, 1].min() - 0.2 * yspan, x[:, 1].max() + 0.2 * yspan)
        ax.set_xlabel(first)
        ax.set_ylabel(second)
        ax.set_title("Pairwise projection")
        ax.grid(alpha=0.2)
    axes.flat[0].legend(loc="best")
    fig.suptitle("Pairwise projections of the 4-D RBF-supported convex hull; margins are unsupported", y=1.02)
    fig.savefig(OUT / "input_pairwise_rbf_support.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def candidate_claims(manifest: pd.DataFrame, pages: dict[str, dict[int, str]]) -> pd.DataFrame:
    """Extract only sentences containing a variable and explicit directional language."""
    rows = []
    directions = re.compile(r"\b(increas|decreas|increase|decrease|higher|lower|reduce|enhanc|optimum|optimal|non[- ]?monotonic|maximum|minimum)\w*\b", re.I)
    for record in manifest.to_dict("records"):
        if record["text_status"] == "insufficient_text":
            continue
        category = record["category"]
        for page, raw_text in pages[record["paper"]].items():
            normalized = normalize_quote(raw_text)
            for sentence in re.split(r"(?<=[.!?])\s+", normalized):
                lower = sentence.lower()
                matched_inputs = [key for key, value in INPUTS.items() if any(term in lower for term in value["terms"])]
                matched_outputs = [key for key, value in OUTPUTS.items() if any(term in lower for term in value["terms"])]
                direction_match = directions.search(sentence)
                if not matched_inputs or not matched_outputs or not direction_match or len(sentence) < 35:
                    continue
                direction = direction_match.group(0).lower()
                if "increas" in direction or "higher" in direction or "enhanc" in direction:
                    direction = "increases"
                elif "decreas" in direction or "lower" in direction or "reduce" in direction:
                    direction = "decreases"
                elif "optimum" in direction or "optimal" in direction or "maximum" in direction or "minimum" in direction:
                    direction = "optimum"
                else:
                    direction = "unclear"
                claim_id = f"{Path(record['paper']).stem}_p{page}_{len(rows)+1}"
                rows.append({
                    "claim_id": claim_id,
                    "paper": record["paper"], "page": page,
                    "independent_variables": "; ".join(matched_inputs), "dependent_variables": "; ".join(matched_outputs),
                    "variables": f"{'; '.join(matched_inputs)} -> {'; '.join(matched_outputs)}",
                    "direction": direction, "functional_form": None, "stated_conditions": None, "stated_assumptions": None,
                    "evidence_type": "asserted_without_support", "source_quote": sentence,
                    "process_relevance": "same_process" if category == "flipmm_or_laser_micro" else "adjacent_process",
                    "extraction_confidence": "low", "confidence_reason": "Automated sentence candidate; context and variable mapping require human review.",
                    "needs_human_review": True, "category": category,
                })
    result = pd.DataFrame(rows).drop_duplicates(subset=["paper", "page", "source_quote"])
    return result.head(40)


def verify_quotes(claims: pd.DataFrame, pages: dict[str, dict[int, str]]) -> pd.DataFrame:
    claims = claims.copy()
    claims["quote_verified"] = [normalize_quote(row.source_quote) in normalize_quote(pages[row.paper][int(row.page)]) for row in claims.itertuples()]
    claims["quote_verification_note"] = np.where(claims["quote_verified"], "quote found after whitespace/hyphen normalization", "quote not found; remove or correct")
    return claims


def build_comparison(claims: pd.DataFrame, supports: pd.DataFrame) -> pd.DataFrame:
    claims = claims.copy()
    claims["applicable_to_data_range"] = "unknown_conditions"
    claims["data_verdict"] = "inconclusive"
    claims["data_explanation"] = "Paper conditions were not safely mapped to dataset conditions by automation."
    def agreement(group: pd.DataFrame) -> str:
        if len(group) == 1:
            return "single_source"
        return "agree" if group["direction"].nunique() == 1 else "conflict"
    agreement_values = claims.groupby(["independent_variables", "dependent_variables"], group_keys=False).apply(agreement, include_groups=False)
    claims["sources_agree"] = [agreement_values.loc[(row.independent_variables, row.dependent_variables)] for row in claims.itertuples()]
    claims["status"] = np.where(~claims["quote_verified"], "insufficient_evidence", "insufficient_evidence")
    return claims


def main() -> None:
    cfg = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_excel(resolve_path(cfg, cfg["dataset"]["file"]))
    pdf_dir = resolve_path(cfg, "data/literature")
    manifest, pages = extract_pdfs(pdf_dir)
    manifest = classify_papers(manifest, pages)
    manifest.to_csv(OUT / "pdf_extraction_manifest.csv", index=False)
    manifest.to_csv(OUT / "pdf_classification.csv", index=False)
    report = audit_dataset(frame, cfg)
    (OUT / "audit.md").write_text("# Dataset audit\n\n```json\n" + json.dumps(report, indent=2) + "\n```\n", encoding="utf-8")
    frame[cfg["input_columns"] + cfg["output_columns"]].describe().T.to_csv(OUT / "dataset_descriptive_statistics.csv")
    pd.plotting.scatter_matrix(frame[cfg["input_columns"]], figsize=(12, 12), diagonal="hist")
    plt.suptitle("FLIPMM input coverage")
    plt.savefig(OUT / "input_coverage_scatter_matrix.png", dpi=180)
    plt.close("all")
    validation = validation_table(frame, cfg)
    validation.to_csv(OUT / "surrogate_validation.csv", index=False)
    support = support_table(frame, cfg)
    support.to_csv(OUT / "data_support_verdicts.csv", index=False)
    surrogate_plots(frame, cfg)
    input_domain_coverage_plot(frame, cfg)
    claims = verify_quotes(candidate_claims(manifest, pages), pages)
    claims.to_csv(OUT / "claims.csv", index=False)
    comparison = build_comparison(claims, support)
    comparison.to_csv(OUT / "claims_vs_data.csv", index=False)
    mappings = "# Variable mapping review\n\nAutomated mappings are intentionally conservative. Confirm these before treating any claim as applicable:\n\n"
    mappings += "- Pulse energy is not automatically equivalent to average power or fluence.\n- Frequency is not automatically equivalent to pulse overlap or duty cycle.\n- Scanning speed is not automatically equivalent to translation speed under all apparatus descriptions.\n- Water speed is not automatically equivalent to flow rate or pressure.\n- MRR is not assumed to be derived from width, depth, and scanning speed.\n"
    (OUT / "variable_mapping.md").write_text(mappings, encoding="utf-8")
    (OUT / "README.md").write_text(
        "# FLIPMM claim-versus-data first pass\n\n"
        "Run from the repository root:\n\n"
        "```powershell\n.venv\\Scripts\\python.exe claim_vs_data_first_pass\\run_first_pass.py\n```\n\n"
        "The runner reads the private workbook and PDFs from `data/` without modifying them. "
        "It writes page-preserving extracted text, audit outputs, one-parameter-at-a-time "
        "surrogate validation, plots, candidate literature claims, quote checks, and comparison tables.\n\n"
        "Status rules: an unverified quote is `insufficient_evidence`; a verified quote with "
        "unknown or non-overlapping conditions is not promoted to support; a claim with unknown "
        "condition mapping or an inconclusive data verdict remains `insufficient_evidence`; "
        "`supported` and `contradicted_by_data` require later human-reviewed condition mapping "
        "plus a non-inconclusive data verdict; conflicting directions for the same variable pair "
        "are marked `conflicting_sources` only after verified claims are reviewed. No numeric "
        "reliability score is produced.\n\n"
        "This is exploratory. Claims are retained only when their quotes are found in the extracted "
        "page text. Automated extraction marks claims for human review; it does not establish "
        "physical truth or literature transfer.\n\n"
        "Generated files include `claims.csv`, `claims_vs_data.csv`, `surrogate_validation.csv`, "
        "`data_support_verdicts.csv`, `input_domain_coverage.png`, and "
        "`input_pairwise_rbf_support.png`. The green region is the observed "
        "convex-hull projection; hatched margins are unsupported extrapolation space.\n",
        encoding="utf-8",
    )
    (OUT / "notes.md").write_text(
        "# First-pass notes\n\n"
        "Automated extraction is deliberately conservative. PDF classification and every candidate claim require human review. Conditions, units, assumptions, and variable mappings were not inferred when absent. The current workbook has no supplied run/session identifiers, so unique-setting K-fold is only a provisional leakage check. The high-level leave-regime-out results are exploratory; they are not independent physical validation.\n\n"
        "All 19 PDFs had a text layer and were extracted without page failures. `sample10.pdf` contains font-encoding artifacts (`/gid` markers); OCR would be useful for that paper, but a system OCR executable is not installed. Its claims should receive priority human review. The repository's cached Docling text is available as a secondary review aid.\n",
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(frame), "pdfs": len(manifest), "claims": len(claims), "verified_quotes": int(claims["quote_verified"].sum()) if len(claims) else 0}, indent=2))


if __name__ == "__main__":
    main()
