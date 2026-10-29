# FLIPMM claim-versus-data first pass

Run from the repository root:

```powershell
.venv\Scripts\python.exe claim_vs_data_first_pass\run_first_pass.py
```

The runner reads the private workbook and PDFs from `data/` without modifying them. It writes page-preserving extracted text, audit outputs, one-parameter-at-a-time surrogate validation, plots, candidate literature claims, quote checks, and comparison tables.

Status rules: an unverified quote is `insufficient_evidence`; a verified quote with unknown or non-overlapping conditions is not promoted to support; a claim with unknown condition mapping or an inconclusive data verdict remains `insufficient_evidence`; `supported` and `contradicted_by_data` require later human-reviewed condition mapping plus a non-inconclusive data verdict; conflicting directions for the same variable pair are marked `conflicting_sources` only after verified claims are reviewed. No numeric reliability score is produced.

This is exploratory. Claims are retained only when their quotes are found in the extracted page text. Automated extraction marks claims for human review; it does not establish physical truth or literature transfer.

Generated files include `claims.csv`, `claims_vs_data.csv`, `surrogate_validation.csv`, `data_support_verdicts.csv`, `input_domain_coverage.png`, and `input_pairwise_rbf_support.png`. The green region is the observed convex-hull projection; hatched margins are unsupported extrapolation space.
