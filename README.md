# Asset Correlation Project

This project computes correlations between asset time series (returns/prices).

Quick start

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
#deactivate to quit venv
pip install -r requirements.txt
```

2. Run the CLI on the sample data:

```bash
python -m src.corr_assets.cli data/sample_assets.csv --method pearson --out out/corr.csv --plot out/corr.png
```

Files

- `src/corr_assets`: package with core functions and CLI
- `data/sample_assets.csv`: small sample dataset
- `tests`: unit tests (run with `pytest`)

# Clean data project
```bash
deactivate
#install cleanpy
pip install cleanpy
cleanpy .
#clean cache from pip
pip cache purge
```
```