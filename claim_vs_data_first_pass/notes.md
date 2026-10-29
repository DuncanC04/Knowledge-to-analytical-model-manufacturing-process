# First-pass notes

Automated extraction is deliberately conservative. PDF classification and every candidate claim require human review. Conditions, units, assumptions, and variable mappings were not inferred when absent. The current workbook has no supplied run/session identifiers, so unique-setting K-fold is only a provisional leakage check. The high-level leave-regime-out results are exploratory; they are not independent physical validation.

All 19 PDFs had a text layer and were extracted without page failures. `sample10.pdf` contains font-encoding artifacts (`/gid` markers); OCR would be useful for that paper, but a system OCR executable is not installed. Its claims should receive priority human review. The repository's cached Docling text is available as a secondary review aid.
