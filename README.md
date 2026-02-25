# MLOps Batch Pipeline — Rolling Mean Signal Generator

A minimal MLOps-style batch job that computes a binary trading signal
from OHLCV price data using a rolling mean strategy.

## What it does

1. Loads config from `config.yaml` (seed, window, version)
2. Reads 10,000 rows of OHLCV data from `data.csv`
3. Computes a rolling mean on the `close` column
4. Generates a binary signal: `1` if `close > rolling_mean`, else `0`
5. Writes structured metrics to `metrics.json` and detailed logs to `run.log`

> The first `window - 1` rows (4 rows with default window=5) are excluded
> from signal computation as they do not have a full rolling window.
> `rows_processed` reflects only rows where a signal was computed (9,996).

---

## Local Setup

**Requirements:** Python 3.9+
```bash
# 1. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the pipeline
python run.py --input data.csv --config config.yaml --output metrics.json --log-file run.log
```

---

## Docker

**Requirements:** Docker installed and running
```bash
# Build the image
docker build -t mlops-task .

# Run the container
docker run --rm mlops-task
```

The container includes `data.csv` and `config.yaml`. Metrics JSON is
printed to stdout and written inside the container.

---

## Configuration

`config.yaml` controls all pipeline parameters:
```yaml
seed: 42      # NumPy random seed for reproducibility
window: 5     # Rolling mean window size
version: "v1" # Pipeline version tag
```

---

## Example Output

`metrics.json` after a successful run:
```json
{
  "version": "v1",
  "rows_processed": 9996,
  "metric": "signal_rate",
  "value": 0.4991,
  "latency_ms": 39,
  "seed": 42,
  "status": "success"
}
```

---

## File Structure
```
mlops-task/
├── run.py            # Pipeline entrypoint
├── config.yaml       # Pipeline configuration
├── data.csv          # Input OHLCV dataset (10,000 rows)
├── requirements.txt  # Pinned Python dependencies
├── Dockerfile        # Container definition
├── metrics.json      # Sample output from successful run
├── run.log           # Sample log from successful run
└── README.md         # This file
```

---

## Error Handling

All errors are caught and written to `metrics.json` with `status: error`:
```json
{
  "version": "v1",
  "status": "error",
  "error_message": "Description of what went wrong",
  "latency_ms": 9
}
```

Handles: missing input file, invalid CSV format, empty file,
missing `close` column, invalid config structure.