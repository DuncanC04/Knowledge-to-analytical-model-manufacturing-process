# Diverse committee profile

Run the opt-in profile explicitly:

From PowerShell, run it through the project environment:

```powershell
uv run python -m src.active_learning --config config/config_diverse_committee.yaml --target-output MRR
```

Alternatively, activate the environment first:

```powershell
.\.venv\Scripts\Activate.ps1
python -m src.active_learning --config config/config_diverse_committee.yaml --target-output MRR
```

Do not use the system `C:\Python312\python.exe`; this project requires
Python 3.13 and its scikit-learn installation is in `.venv`.

This profile writes to `output/diverse_committee/` and uses the new
`diverse_accuracy` selector. The default [config.yaml](./config.yaml), legacy
selection rules, and previous output directories are unchanged.

The selector keeps candidates within an adjusted-R² tolerance of the best
development candidate, then greedily adds candidates that are farthest apart
in validation-prediction space. Validation predictions are stored in candidate
archives for auditability; the locked test set is not used for selection.
