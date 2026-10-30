# Two-Stage System 1 Results — Chunk-Level Run

Source: [`output/system1_labels.json`](../../output/system1_labels.json).

## One-slide summary

The two-stage pipeline split 19 papers into 253 sentence-aware chunks, screened
each chunk with Laya on the GPU, retained 249 chunks, and generated an
LLM-based description for every retained chunk.

| Metric | Result |
|---|---:|
| Source papers | 19 |
| Chunks screened | 253 |
| Chunks retained | 249 / 253 (98.4%) |
| Chunks rejected | 4 / 253 (1.6%) |
| Retained chunks with LLM descriptions | 249 / 249 |
| Chunk size / overlap | 800 / 120 tokens |
| Laya positive-label threshold | `noul > 0.5` |
| Description model | `gpt-4o-mini` |
| Laya device | NVIDIA RTX 3060 Ti, CUDA 12.8 |

## Second-stage LLM results

![Second-stage LLM evidence descriptions](../../output/system1_stage2_results.png)

- The LLM generated descriptions for all 249 Laya-retained chunks.
- A concise summary was produced for 100% of retained chunks.
- A relationship statement was present in 90.4% of descriptions.
- An evidence quote was present in 92.8% of descriptions, helping trace the
  description back to the source chunk.
- Mechanism information was present in 85.1% of descriptions.
- Operating conditions were present in 73.1% of descriptions.
- Explicit ranges were present in 43.4% of descriptions, making ranges the
  least consistently populated field.
- Variables were explicitly named in 50.6% of descriptions and should be
  improved before automated claim comparison or equation fitting.

The visual is also available as
[`system1_stage2_results.png`](../../output/system1_stage2_results.png).

## Laya screening results

| Question | Positive chunks |
|---|---:|
| Physical relationship | 229 / 253 (90.5%) |
| Mathematical relationship | 240 / 253 (94.9%) |
| Physical mechanism | 244 / 253 (96.4%) |
| Quantitative evidence | 223 / 253 (88.1%) |
| Operating conditions | 233 / 253 (92.1%) |

The aggregate score is the fraction of the five screening questions answered
positively:

```text
0.0: 4   0.2: 2   0.4: 9   0.6: 11   0.8: 19   1.0: 208
```

## Structured relationship findings

Among the 249 retained chunks:

| Extracted feature | Count |
|---|---:|
| Comparative effect | 142 |
| Empirical fit/model | 48 |
| Causal/mechanistic | 28 |
| Equation/formula | 12 |
| Threshold/nonlinear | 10 |
| Relationship form not stated | 9 |
| Nonlinear/nonmonotonic direction | 198 |
| Direction not stated | 19 |
| Increasing direction | 14 |
| Decreasing direction | 12 |
| Conditional/interaction direction | 2 |
| Simulation/model evidence | 197 |
| Measured experimental evidence | 31 |
| Theoretical/derived evidence | 16 |
| Fitted-from-data evidence | 4 |

The per-question confidence distributions are shown in
[`system1_noul_confidence_boxplot.png`](../../output/system1_noul_confidence_boxplot.png).
The orange line is the median and the black diamond is the mean.

Range and applicability signals:

- Explicit range present: 134 / 249 (53.8%).
- Fixed operating conditions present: 187 / 249 (75.1%).
- Applicability limit, threshold, transition, or caveat present: 171 / 249 (68.7%).

## Examples to present

### Example 1 — magnetic-field-assisted surface quality

Chunk: `sample2.pdf:chunk-158`

- Laya score: 1.0 / 1.0.
- Classified as causal/mechanistic with a conditional or interaction effect.
- The LLM description identified a repulsive magnetic field, Lorentz-force
  effects on charged-particle motion, improved surface quality, and a reported
  reduction in micro-groove line width.

Presentation point: the chunk-level system can isolate a specific mechanism
from a larger paper instead of treating the whole paper as one label.

### Example 2 — liquid-medium comparison

Chunk: `sample10.pdf:chunk-26`

- Laya score: 0.4 / 1.0, retained under the current “at least one positive
  label” rule.
- The LLM description compared microchannel width and heat-affected zones
  across ethanol, deionized water, and phosphoric-acid solutions.

Presentation point: low-score chunks can still contain useful comparative
evidence, but they should be reviewed before being used for equation fitting.

### Example 3 — process-parameter and condition evidence

Chunk: `sample2.pdf:chunk-154`

- Laya score: 1.0 / 1.0.
- The LLM description identified femtosecond-laser processing, peak power
  density above approximately `10^12 W/cm²`, ethanol as the medium, and
  controlled laser/motion parameters.

Presentation point: the second stage turns structured screening signals into a
short, human-readable record of variables, conditions, and process relevance.

### Example 4 — rejected chunks

Four chunks received a score of 0.0 and were rejected. They did not trigger any
of the five positive Laya screening labels and therefore did not consume a
second-stage LLM description call.

## Suggested talk track

> We moved from paper-level classification to chunk-level classification. The
> 19 papers produced 253 sentence-aware chunks, allowing the system to isolate
> individual mechanisms, comparisons, equations, and operating conditions. Laya
> provides consistent structured screening, while the LLM summarizes only the
> retained evidence. The current run retained 98.4% of chunks, which is useful
> for recall but indicates that the filter is still permissive and should be
> calibrated before analytical-model fitting.

## Important caveats

1. This is a chunk-level retrieval/extraction result, not a precision/recall
   benchmark. The previous 19-record paper-level result has been superseded for
   this purpose.
2. The retention rule is intentionally permissive: one positive label retains a
   chunk. The 98.4% retention rate suggests testing stricter policies, such as
   requiring a physical or mathematical relationship, or at least two positive
   labels.
3. 242 of 253 Laya responses reported truncation metadata. The chunk size is
   now controlled, but the multi-question request still needs a follow-up pass
   with fewer questions per call or a verified windowing strategy.
4. Numeric range extraction is only a review aid. Candidate ranges must be
   checked against the original chunk before being used in a model.
5. The Laya checkpoint reported uncalibrated confidence temperatures. The
   labels use the explicit `noul > 0.5` rule; confidence values should not yet
   be treated as calibrated probabilities.
