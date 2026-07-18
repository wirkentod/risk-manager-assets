# Risk Management Asset Project

This project computes risk assets and portfolios.

Quick start

1. **Clone the repository and navigate to the directory:**

```bash
git clone https://github.com/wirkentod/risk-manager-assets.git
cd risk-manager-assets
```

2. **Create a virtual environment and install dependencies:**

```bash
# On macOS/Linux:
python -m venv .venv
source .venv/bin/activate

# On Windows (PowerShell):
.venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```
## 📊 Command-Line Interface (CLI) Usage
3. Run the CLI on the sample data:

Run the analysis tool directly by invoking the properly mapped module:

```bash
python -m src.pyfolio.cli data/sample_assets.csv --term 1A --dailychange log --method pearson --out out/corr.csv --plot out/corr.png --plotfolio out/pflio
```

### Available Arguments:
* `--term`: Timeframe window for the analysis (e.g., `1Y` for one year, `6M` for six months).
* `--dailychange`: Formula for financial return calculation (`log` for logarithmic returns, `simple` for standard percentage change).
* `--method`: Statistical correlation method (`pearson`, `spearman`, or `kendall`).
* `--out`: Output destination path to save the generated correlation matrix as a CSV file.
* `--plot`: Output destination path to export the heatmap chart imagery.

## 📁 Repository Structure
* `src/pyfolio/` - Core source code package and CLI entrypoint logic.
* `src/pyfolio/config/` - System-wide global configurations and parameters.
* `data/` - Storage directory for static datasets (contains `sample_assets.csv`).
* `tests/` - Unit tests folder suite.

## 🧪 Running Tests
To ensure the statistical calculations are running consistently, execute the test suite locally:

```bash
pytest
```