# `data/` — put your inputs here

This folder holds the two kinds of input the pipeline needs. Its contents are
git-ignored (so large/private files are never committed); only this README and
the empty folder markers are tracked.

```
data/
├── <your-dataset>.xlsx     # the experimental dataset (one row per experiment)
├── literature/             # the source PDFs that knowledge is extracted from
│   ├── sample1.pdf
│   └── sample2.pdf
└── cache/                  # auto-generated parsed-PDF cache (do not edit)
```

## 1. Dataset spreadsheet
An Excel file with one column per variable. The column headers must match the
`column:` fields in `config/config.yaml` **exactly** (including units and
spaces). Point `dataset.file` in the config at this file.

## 2. Literature PDFs
Drop the papers you want to mine into `data/literature/` and list them under
`knowledge.pdf_files` in the config. They are parsed once and cached to
`data/cache/parsed_data.pkl`; delete that file if you change the PDFs.
