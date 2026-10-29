# FLIPMM: literature-grounded diagnosis of extrapolation reliability

## Research objective and scope

Working question: **Does condition-specific literature evidence improve detection of unreliable single-parameter extrapolations from fitted analytical models, beyond experimental distance, parameter uncertainty, and candidate-model disagreement?**

Secondary question: Can verified evidence distinguish actionable limitations in parameter identification, competing equations, and assumption applicability? Explanations are diagnoses with evidence, not automatically proven causes.

Inputs provided: laser pulse energy, frequency, scanning speed, water speed. Outputs provided: channel width, channel depth, material removal rate (MRR), heat-affected zone (HAZ). There are reportedly 19 PDFs, potentially spanning FLIPMM, MSLA, and TADCR. Their identities, relevance, operating ranges, and completeness remain unverified. Private measurements were not supplied. This plan does not presume numerical ranges, units, row counts, independent replicates, or experimental provenance.

Primary pilot: HAZ under high-energy extrapolation, subject to adequate levels and overlap. HAZ is chosen as a focused thermal-response target, not because a failure has been established. Freeze this choice before revealing held-out outcomes; if infeasible, document a feasibility-based replacement. Expand to depth, width and MRR only after the pilot; do not count correlated outputs as independent experiments. In particular, establish whether MRR is computed from width, depth and scanning speed.

Hypothesis H1: applicability evidence improves ranking of failed extrapolations beyond numerical diagnostics. H2: the gain is concentrated in cases with genuinely informative condition/mechanism evidence. H3: inappropriate evidence can worsen assessments; admitting insufficient evidence is necessary. A null result is scientifically useful and must be reported.

## 1. Audit and freeze the study inputs

Create a data dictionary mapping actual spreadsheet columns to the canonical names in the implementation. Record units, material, pulse duration, wavelength, spot size, focusing geometry, water conditions, passes, instrument, measurement procedure and specimen/run/source identifiers. Do not infer that any particular metadata are available. Distinguish measured responses, averaged replicates, response-surface predictions and previously interpolated values. If the table is already surrogate-generated, label the entire retrospective evaluation as model-on-model rather than independent physical validation.

Audit unique input levels, repeated settings, missing values, rank, correlations, output zeros, negative values, feasible combinations and coverage. Obtain raw replicates if possible. Exact repeated settings stay together; supplied group identifiers may connect additional observations. Study/run grouping can eliminate usable train/test comparisons; report that rather than splitting dependent observations to preserve sample size.

Predefine a practically meaningful absolute tolerance tau for each output, in its real units, with the advisor. If no engineering tolerance exists, use a sensitivity grid normalized by a development-only output scale; call these statistical thresholds. Define failure as absolute error exceeding tau. Define an overconfidence event separately: a prediction interval misses its outcome while its width is below a predeclared useful-width bound. Neither R-squared nor 1 minus R-squared is a percent physical error.

Freeze checksums for data, PDF corpus, split manifests, source lineage, extraction versions and model versions. The final evaluation outcomes must not be supplied to model generators, evidence assessors or threshold selection.

## 2. Partition the 19 PDFs before extraction

Inventory every PDF with SHA-256, title, DOI if known, process tags, material, laser regime, input/output relevance, page-text availability and provenance role. Tags: direct FLIPMM; related laser/liquid process; other process; uncertain. Related studies are not automatically transferable. Retain irrelevant papers as retrieval distractors, but do not count them as applicable evidence.

Source roles: admissible external evidence; target-dataset source; duplicated/reproduced target evidence; evaluation-report source; unresolved lineage. Exclude target outcomes and fitted response surfaces revealing those outcomes from the primary evidence pool. Keep an explicit separate external-data-transfer arm if numerical external observations are used. A source containing evaluation outcomes may be excluded entirely, or masked under an audited procedure fixed before extraction. Corpus filtering must not use test prediction errors.

Two reviewers independently annotate a stratified passage subset, including irrelevant passages, exceptions, ranges, equations, mixed conditions and unsupported claims. Preserve disagreements and adjudication. Domain expertise is needed for physical applicability. Exact quote validation is necessary but insufficient for scientific validity.

## 3. Define single-axis extrapolation tasks

For each input j, sort its distinct operating levels. Hold out one or more complete highest levels (and later lowest levels), preserving at least two retained levels, enough fitting rows, and enough independent groups. Use level boundaries, not random row splits or quantiles that divide tied settings. Primary study changes only one boundary at a time. The original multi-axis filtering can be a secondary replication, not the main design.

The retained side contains fitting and inner validation/calibration groups. The outside levels are test-only. Report the distance from the retained boundary divided by retained span. Also report nearest-neighbor distance in scaled joint input space and other-axis range violations. Marginal bounds alone do not prove joint support. Where possible use matched non-extrapolated settings or predefined overlapping strata. Exclude or separately label multiaxis shifts. If matching fails, the experiment cannot isolate the requested axis.

Nested development: outer evaluation groups/cases remain untouched; inner grouped folds choose models, hyperparameters, interval settings and diagnostic thresholds. If using 30 fitting rows to emulate the original scarcity setting, subsample whole groups where feasible and retain a documented inner-validation budget. Treat 30 as a design option, not a known dataset count. Compare larger fitting budgets when data permit.

Several boundaries and random fitting subsets of the same dataset are dependent. Aggregate paired differences within task/source blocks and report variability across boundaries without pretending these are independent datasets. With one physical dataset, broad generality remains unestablished.

## 4. Use the advisor's RBF proposal in two distinct roles

### A. Development-only surrogate and densification

Fit an RBF to the outer development rows only, scaling inputs and outputs using each inner training fold. Choose kernel and smoothing by grouped cross-validation. Initial grid: linear, thin-plate spline and cubic kernels; positive smoothing 0.001, 0.1, 1 on standardized outputs. Check rank, physical positivity, held-in residuals and boundary errors. Hyperparameter sensitivity is mandatory.

Generate Latin-hypercube candidate inputs inside the development convex hull, rejecting infeasible combinations using domain rules. The supplied code implements convex-hull filtering; equipment/feasibility constraints require a user-supplied screen before scientific use. Never silently clip negative responses. These points are synthetic and share their parent's information. They can test code or train a student model in a separate augmentation arm; they cannot enlarge the independent sample size, calibrate a physical interval, or validate predictions against the same generating RBF.

The learner must not receive the RBF's gradients or withheld outputs unless that is an explicitly declared information condition. Do not present RBF gradients as literature evidence.

### B. Controlled teacher-world benchmark

For method development, an RBF may be fitted to a designated reference subset spanning a broader known range. Then hide its outer evaluations from the learner, train the learner in an inner subdomain and test in the teacher's broader supported domain. This is extrapolation for the learner but interpolation for the teacher. A separate real experimental audit subset, never used to fit or select the teacher, assesses teacher fidelity. If no audit subset is feasible, the benchmark remains an unvalidated synthetic world.

Use multiple teacher specifications and, when affordable, grouped-resampling teacher fits. Their spread is sensitivity to the teacher construction, not a guaranteed physical uncertainty interval. Do not extrapolate a full-data RBF beyond the actual measured envelope and call it ground truth.

Controlled perturbation worlds can add a smooth response change beyond a selected boundary, omitted interactions, noisy observations or mismatched evidence. Label all perturbations invented; no claim that they represent actual FLIPMM transitions. Benchmark evidence for these worlds must be explicitly generated benchmark metadata, never fabricated quotations attributed to the 19 papers. Use this arm to test diagnostic recoverability; assess scientific literature separately on real observations.

## 5. Candidate models and interval baselines

Preserve Kiarash's pipeline unchanged as a reference. For the extension, save every retained candidate's equation, coefficient definitions, bounds, fitting failures, units, assumptions, retrieval sources, generation seed and validation score. Freeze generation/refinement budgets. Retain diverse adequate fits; deduplicate equivalent or nearly identical prediction functions. Do not assume analytical form establishes physical correctness.

Use simple linear/quadratic response surfaces and a probabilistic regressor as non-LLM reference models. For Kiarash-style candidates, start with grouped bootstrap refitting of coefficients and a finite diverse candidate ensemble. Add development-only observation-noise estimates where available. Selection-aware resampling that reruns model search is a later sensitivity arm; coefficient-only bootstraps omit search uncertainty and must be labeled accordingly. Ensemble intervals are empirical candidates, not certified extrapolation intervals. Conformal intervals can be a reference, but ordinary exchangeability-based validity must not be claimed after domain shift.

Do not execute unreviewed LLM Python in the main process. Prefer a restricted expression representation or isolated reviewed execution with resource limits. Finite model-pool probability is not probability that the physical truth lies in the pool. Literature-derived coefficient priors are optional later work, limited to coefficients with matching scientific definitions.

## 6. Literature evidence representation and primary diagnostic

Claim fields: claim ID, source and page, exact quote, process/material context, parameter, target, relationship type/direction, numerical range and units, equation/assumption link, observed versus inferred status, applicability classification, reviewer status, and lineage role. Explicit nulls denote unreported ranges. Do not convert absent ranges into universal applicability.

Claim validation checks source text, IDs, allowed parameters/targets, units and ranges. Human review checks semantic fidelity and physical transfer. Build evidence at a candidate-model/query level: supported assumption, challenged assumption, possible regime transition, conflicting evidence, or insufficient information. Keep source independence and contextual match visible. Numerical overlap is necessary only when relevant, and never sufficient alone.

Begin with expert-selected, verified passages. Primary comparison holds candidate predictions fixed:
A: numerical distance alone.
B: distance + parameter uncertainty + model disagreement + inner-validation diagnostics.
C: B + verified applicability diagnostics.
D, later: B + automatically retrieved/extracted diagnostics.

A simple prespecified ordinal evidence warning is acceptable for the pilot: supported / unresolved / challenged, with passage-based justification. Evaluate it as a ranking feature, not a probability. If learning a logistic risk mapper, fit on separate development tasks with standardization, regularization and calibration chosen there; preserve untouched outer tasks. With inadequate independent tasks, do not fit a confidence model. Do not add arbitrary literature and numerical scores to claim total confidence.

## 7. Tests and metrics

Primary endpoint: paired improvement of B versus C in failure detection, measured with average precision (report failure prevalence) and risk-coverage curves. Predeclare the main acceptance fraction, for example 50%, before test inspection; report selective MAE and failure rate at that fraction plus full curves. AUROC is secondary and undefined for single-class tasks. Always compare the retained cases and abstention burden.

For predictive intervals report nominal level, empirical coverage, mean/median width and proper interval score. A wider interval alone is not a success. For probabilistic failure estimates, evaluate Brier score and reliability diagrams only if a held-out risk model actually produces probabilities. A rank warning is not calibrated confidence.

Secondary diagnostic endpoints: evidence extraction precision/recall; quote and provenance correctness; reviewer agreement; mechanism/assumption diagnosis on controlled worlds; targeted intervention benefit. Evaluate interventions such as adding boundary measurements, adding replicates, relaxing a prior or expanding a mechanism family against equal-budget alternatives. Improvement in sensitivity after changing a component does not automatically prove physical causation.

Report paired block-bootstrap intervals only when there are enough genuinely independent blocks; no row bootstrap across repeated settings and no p-values treating thousands of RBF points as independent experiments. If independence is weak, report descriptive paired effects and reserve confirmatory conclusions for new experiments or additional independent systems.

## 8. Ablations and failure modes

Required ablations: no literature; correct matched evidence; same-volume irrelevant evidence; imperfectly matched evidence; contradiction/exception evidence; expert versus automated extraction; fixed predictions versus literature-modified predictions. A negative control may permute evidence assignments across tasks, but preserve valid source text and disclose the manipulation. Re-run within the development protocol, never choose controls based on favorable final results.

Important failure modes: all candidates share missing physics; literature gives only trends rather than magnitudes; current-range prior agreement conceals future breakdown; target-source leakage; already synthetic source data; preprocessing leakage; four-dimensional sampling outside feasible support; inconsistent units; MRR-induced output dependence; few distinct levels; no evidence variation; and overfitting the diagnostic to one process.

## 9. Sequential go/no-go milestones

Weeks 1-2: private data and corpus audit; freeze task feasibility, output tolerance and source exclusions. Deliver data dictionary, corpus manifest and split registry. No numerical ranges may be invented.
Weeks 3-4: reproduce reference fitting, establish numerical UQ baselines and audit the RBF. Deliver real-holdout failure maps and separate synthetic benchmarks. If useful failures or adequate support do not exist, revise the task transparently before the main study.
Weeks 5-6: expert evidence extraction and blinded applicability labeling. Deliver validated claims and inter-reviewer results. If applicable evidence is too sparse, study evidence insufficiency or narrow the corpus question; do not invent mechanism boundaries.
Weeks 7-8: frozen-predictor A/B/C comparison; evaluate incremental signal on untouched tasks. Advance automation only if correct evidence provides useful signal or if extraction reliability itself is the revised research objective.
Weeks 9-10: automated retrieval/extraction and ablations. Use ordinary hybrid retrieval first. Add a graph only for demonstrated missed multi-hop evidence, keeping extraction/evaluation fixed.
Weeks 11-12: targeted new experiments if feasible, robustness analysis, error audit and writing. Strong claim requires independent physical validation; otherwise describe a retrospective pilot plus controlled benchmark.

Success means a reproducible added diagnostic benefit, not necessarily improved point prediction. Failure to improve over B is a valid finding. A defensible paper would state which evidence types helped, where the diagnostic failed, and the limits of confidence attribution. Novelty is a hypothesis requiring continued related-work review, not established by this plan.

## 10. Implementation status and acceptance criteria

Included runnable commands: private-table audit, group-safe single-axis splits, development-only RBF fitting with grouped CV, convex-hull synthetic generation, exact-quote/provenance claim checks, and external-prediction evaluation. Included tests verify test-label invariance of the RBF and group/boundary integrity. Demo measurements are invented dimensionless software fixtures.

Not implemented or claimed complete: authenticated LLM calls, automated PDF extraction, original candidate generation integration, Bayesian fitting, interval construction, learned risk calibration, physical feasibility screening, expert annotations, teacher-world perturbation experiments, independent inferential statistics or real FLIPMM results. These are explicit work packages above. This scaffold provides interfaces so those components can be added without changing the evaluation contract.

Acceptance criteria before physical claims: verified data provenance and units; sufficient independent splits; excluded target-outcome sources; reviewed executable candidates; predeclared tolerance and evaluation; expert-verified evidence; untouched physical test results; and honest reporting of dependence and interval limitations.

## References and starting points

- Kiarash project and original baseline: https://github.com/kiarash99Naghavi/Knowledge-to-analytical-model-manufacturing-process
- SciPy RBFInterpolator API and smoothing/kernel definitions: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html
- Model Discovery Agent, finite-pool uncertainty limitation: https://arxiv.org/abs/2608.09696
- Applicability-domain diagnostics for unreliable uncertainty: https://doi.org/10.1021/acsomega.5c11875
- Distribird, optional future coefficient-prior module: https://github.com/HUN-REN-AI1Science/Distribird

Methods in this plan are proposed study choices. No physical relationship, range, PDF relevance or empirical gain is inferred from the software demo.

## 11. Added work package: disagreement, physical constraints and active learning

### Purpose and relationship to the primary study

Add an explicit investigation of disagreement rather than treating ensemble spread as a single fixed baseline. Separate two questions: (D1) Does disagreement identify unreliable extrapolations? (D2) Does querying high-disagreement points improve predictions efficiently? A useful acquisition function need not be a calibrated uncertainty estimator, and reduced spread after a query need not indicate reduced physical error.

Reference: *Active Learning in Symbolic Regression with Physical Constraints*, arXiv:2305.10379v3, https://arxiv.org/html/2305.10379v3. The paper forms a query-by-committee (QBC) from an accuracy-complexity Pareto frontier, evaluates coefficient-of-variation and information-based pairwise disagreement, and incorporates physical knowledge through soft constraints. Its findings also show that some constraints can hinder discovery. These are precedents/baselines, not proposed novelty. The FLIPMM adaptation below changes the evaluation objective from equation rediscovery to held-out prediction reliability and information-efficient domain expansion.

### 11.1 Committee construction as an experimental factor

Compare committees from the same development-only candidate archive:

1. Top models by grouped inner-validation error, reflecting the existing workflow.
2. Accuracy-complexity Pareto frontier: an equation is retained if no other is at least as accurate and at least as simple, with a strict improvement in one. Define complexity before evaluation, such as operator-weighted expression-tree size; do not compare arbitrary source-code length.
3. Diverse near-optimal committee: candidates must meet a development-only quality threshold, then are selected to span distinct prediction behavior and structural families. Record committee size, exclusions and similarity criteria.

Keep equal committee sizes where feasible, or report a size-matched sensitivity analysis. Pareto membership alone does not guarantee adequate fits: report the effect of an explicit accuracy screen. Deduplicate algebraically equivalent and numerically redundant equations. Do not treat many prompt variants of the same formula as independent scientific support. Run multiple candidate-generation seeds under equal budgets, and distinguish variability from search from variability within a fitted committee. Retain invalid/nonfinite prediction flags as diagnostic outputs rather than silently averaging them away.

### 11.2 Disagreement measures

For predictions f_m(x), m=1,...,M, define mu(x) as their mean and sigma(x) as their population standard deviation. Start with equal weights; validation-weighted variants are secondary and are not Bayesian posterior probabilities unless justified by a separate inference procedure.

- **Absolute spread:** sigma(x), reported in output units.
- **Tolerance-scaled spread:** sigma(x)/tau, allowing comparison with the declared engineering error tolerance.
- **Paper CV baseline:** sigma(x)/mu(x). Report only where predictions have appropriate positive scale and the mean is sufficiently away from zero. Flag undefined or negative-scale cases. A stabilized alternative sigma(x)/max(abs(mu(x)), epsilon) is a separate adaptation, with epsilon fixed using development information; do not label it an exact replication.
- **Paper IBMD baseline:** average over unordered model pairs of log(1 + abs(f_i(x)-f_j(x))/max(f_i(x),f_j(x))). Reproduce only where the pairwise denominators are positive. Zero or negative denominators require explicit invalid flags. A stabilized absolute-denominator variant is a separate sensitivity analysis. Log scaling does not eliminate all effects of pathological candidates.
- **Robust spread:** interquartile range or median absolute deviation of committee predictions, scaled by tau. This tests sensitivity to a few extreme equations.
- **Directional disagreement, exploratory:** disagreement about the sign of each input derivative, evaluated at feasible local perturbations with a development-fixed near-zero threshold. Preserve whether the derivative disagreement concerns the extrapolated parameter. This is an extension, not a method attributed to the paper.

Do not compute standard deviation of actual test errors as an acquisition score: test outcomes are unavailable when choosing a query. Use prediction disagreement before label revelation. Test-error dispersion is a retrospective evaluation quantity.

### 11.3 Separate coefficient and structure contributions

Where grouped bootstrap refits are available, generate coefficient draws within each model structure. Let mu_m(x) be the coefficient-averaged prediction and v_m(x) its within-structure variance. Under a declared finite mixture with fixed normalized weights w_m, the variance of its latent prediction distribution is:

V_within(x) = sum_m w_m v_m(x)

V_between(x) = sum_m w_m [mu_m(x) - sum_l w_l mu_l(x)]^2

V_total(x) = V_within(x) + V_between(x).

This is a conditional decomposition for that mixture. Grouped bootstrap distributions are not automatically Bayesian posteriors; observation noise is separate; missing mechanisms outside the candidate family are not represented. Compare which component predicts held-out error and which decreases after additional measurements. Do not claim this decomposition identifies every cause of uncertainty.

### 11.4 Physical-constraint ablations and literature's role

Compare unconstrained candidates, candidates with basic domain constraints verified by the advisor, and candidates with context-specific literature-derived soft constraints. Candidate examples include nonnegative geometric responses and local derivative trends, but no actual FLIPMM trend or global monotonicity is assumed. Distinguish restrictions on predictions from restrictions on feasible input combinations.

For a constrained search or fitting arm use a normalized development loss plus a weighted, domain-restricted constraint-violation penalty. Set/tune penalty strengths using inner development only. Compare weak/strong penalties and, optionally, early-only versus persistent constraint application. The paper's particular penalty strength must not be transferred across different units and objectives without normalization.

Preserve three separate comparisons: (i) constraints modify generation/fitting; (ii) evidence modifies committee membership or weights; (iii) evidence modifies only the assessment. The main fixed-predictor study uses (iii). Explore (i)-(ii) afterward, since changed predictions otherwise confound assessment value.

Measure whether constraints reduce disagreement AND error, reduce disagreement while increasing error, or preserve useful disagreement. Include a controlled mismatched-constraint benchmark with openly labeled synthetic metadata. Particularly important failure: literature restrictions remove the relevant alternative, producing confident agreement on an incorrect mechanism.

### 11.5 Static evaluation: is disagreement informative?

On the frozen real and teacher-world splits, compare disagreement measures and committees against absolute error, tolerance failures and interval misses. Use ranking correlation, average precision with prevalence, selective error/risk-coverage, and disagreement-versus-error plots within distance bands. Report undefined metrics. Use the same held-out inputs for paired comparisons.

Cross-tabulate low/high disagreement with low/high error using thresholds fixed on development cases. Inspect all four quadrants, especially low-disagreement/high-error cases. Blind literature reviewers to test errors and disagreement scores when annotating applicability, preventing outcome-driven explanations. Test whether literature diagnostics add signal within matched disagreement/distance strata. Examine coefficient identifiability, correlated candidates, mismatched evidence and assumption challenges as competing explanations, leaving unresolved cases explicit.

### 11.6 Sequential evaluation: which points should be measured next?

Create three disjoint roles: initial labeled data, an acquisition pool whose labels are hidden until selected, and a locked evaluation set. The acquisition pool may include expanded-domain points, but evaluation rows/groups never enter it. Once queried points extend training support, record the new boundary; distinguish remaining extrapolation from points that have become interpolation. Report performance on a fixed final target region as well as on the remaining out-of-support region.

Compare policies under identical initial data, feasible pools, response budgets and candidate-generation budgets:

- Uniform random feasible acquisition.
- Space-filling/farthest-from-observed acquisition with scaling fitted from available training data.
- QBC using the paper CV and IBMD baselines where valid.
- QBC using tolerance-scaled or robust spread.
- Literature-informed acquisition, as a separately declared extension.

Use one selected operating condition per round initially; a single condition may yield all four outputs, so charge the experimental cost once and define any multi-output acquisition aggregation in advance. Compare target-specific policies first. Later batch policies must discourage selecting redundant neighboring settings. If the acquisition pool is a retrospective table, reveal only selected group responses and call this pool-based replay, not new experimentation. Restrict any optimization over continuous inputs to physically feasible conditions for which the oracle can return labels.

For an initial literature-informed policy, reserve an advisor-fixed fraction of queries for evidence-flagged assumption challenges and use ordinary QBC for the remainder. If a stratum is empty, use a prespecified random feasible fallback. Compare against a same-budget QBC/random mixture to separate the benefit of literature from the benefit of exploration. Select fractions on development tasks only. This is a proposed heuristic, not claimed optimal value-of-information design.

At each round: fit/refit on available labels, regenerate candidates only under the declared policy, compute committee predictions and diagnostics, choose without seeing pool labels, reveal the selected response, update, and evaluate offline against the locked set. Either refit a frozen structure archive in all arms or regenerate structures with equal budgets in all arms; report these as separate experiments. Evaluation errors never trigger model generation or acquisition decisions.

### 11.7 Oracle discipline and endpoints

Use a broader RBF teacher only in the controlled teacher-world branch described in Section 4B. Candidate acquisition and evaluation inputs must remain inside its supported reference region and feasible input domain. A development-only RBF from Section 4A cannot provide validated labels outside development support. RBF oracle labels test learning against that surrogate; physical claims require locked measured responses or subsequent experiments.

Sequential endpoints: area under the held-out error-versus-query curve, failure rate versus query budget, useful interval coverage/width, residual extrapolation performance, and queries required to reach a prespecified tolerance when reached. Use an explicit fixed evaluation schedule and stable-round criterion if defining sustained attainment. Do not stop the active learner based on locked test scores; threshold-crossing counts are retrospective endpoints. Compare selected locations and whether measurements resolve coefficient ambiguity, competing structures or an evidence-flagged boundary. Falling disagreement alone is not success.

Use multiple initialization/search seeds and reference-teacher specifications. Pair seeds and initial sets across policies. These repeats characterize algorithmic variability on the same system, not additional independent physical datasets. Start with a small budget chosen from pool feasibility; do not invent a measurement budget or operating limits for the unavailable data.

### 11.8 Deliverables, prioritization and interpretation

Before adding automation, deliver (1) committee archive with complexity and validation metadata, (2) disagreement maps by parameter and target, (3) static failure-detection comparisons, (4) low-disagreement/high-error case audit, and (5) sequential learning curves with acquisition manifests and oracle provenance.

Integrate committee/spread comparisons into Weeks 3-4. Add constraint/applicability annotation to Weeks 5-6. Run static disagreement and incremental-literature tests during Weeks 7-8. Treat active learning as an optional follow-on phase after static results, requiring roughly four additional weeks or replacement of the retrieval-automation phase; do not assume all expanded work fits the original schedule.

Implementation status: this section is a research-design addition only. The existing code scaffold has not been extended with disagreement or acquisition implementations. No private data were sought for this update. QBC, CV, IBMD and constraint regularization are existing methods. Potential added value is demonstrating when source-grounded applicability evidence distinguishes useful disagreement from shared model failure, and whether acting on that distinction improves FLIPMM domain expansion.
