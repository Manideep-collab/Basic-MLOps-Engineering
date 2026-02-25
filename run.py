"""
MLOps Batch Pipeline — Rolling Mean Signal Generator
Computes a binary trading signal from close price vs rolling mean.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml



# Logging setup
def setup_logging(log_file: str) -> logging.Logger:
    """Configure logger to write to both file and stdout."""
    logger = logging.getLogger("mlops_pipeline")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# Config loading & validation
REQUIRED_CONFIG_KEYS = {"seed", "window", "version"}

def load_config(config_path: str, logger: logging.Logger) -> dict:
    """Parse and validate config YAML. Raises ValueError on bad config."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("Config YAML must be a mapping of key-value pairs.")

    missing = REQUIRED_CONFIG_KEYS - config.keys()
    if missing:
        raise ValueError(f"Config missing required keys: {missing}")

    if not isinstance(config["seed"], int):
        raise ValueError(f"'seed' must be an integer, got: {type(config['seed'])}")
    if not isinstance(config["window"], int) or config["window"] < 1:
        raise ValueError(f"'window' must be a positive integer, got: {config['window']}")
    if not isinstance(config["version"], str):
        raise ValueError(f"'version' must be a string, got: {type(config['version'])}")

    logger.info(
        "Config loaded and validated | version=%s  seed=%d  window=%d",
        config["version"], config["seed"], config["window"],
    )
    return config


# Data loading & validation
def load_data(input_path: str, logger: logging.Logger) -> pd.DataFrame:
    """Load CSV and validate structure. Raises on any data issue."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    try:
        df = pd.read_csv(path)

        # Handle edge case where entire header row is quoted as one string
        if len(df.columns) == 1 and "," in df.columns[0]:
            import io
            raw = path.read_text()
            cleaned = "\n".join(line.strip('"') for line in raw.splitlines())
            df = pd.read_csv(io.StringIO(cleaned))

    except Exception as exc:
        raise ValueError(f"Failed to parse CSV: {exc}") from exc

    if df.empty:
        raise ValueError("Input CSV is empty.")

    df.columns = [c.strip().lower() for c in df.columns]

    if "close" not in df.columns:
        raise ValueError(
            f"Required column 'close' not found. Available columns: {list(df.columns)}"
        )

    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    invalid_rows = df["close"].isna().sum()
    if invalid_rows > 0:
        logger.warning("Dropped %d rows with non-numeric 'close' values.", invalid_rows)
        df = df.dropna(subset=["close"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid numeric 'close' values found after cleaning.")

    logger.info("Data loaded | rows=%d  columns=%s", len(df), list(df.columns))
    return df



# Signal computation
def compute_signals(df: pd.DataFrame, window: int, logger: logging.Logger) -> pd.DataFrame:
    """
    Compute rolling mean and binary signal.

    First (window-1) rows produce NaN rolling_mean and are EXCLUDED
    from signal computation and metrics. This is documented and consistent.
    """
    logger.info("Computing rolling mean | window=%d", window)
    df = df.copy()
    df["rolling_mean"] = df["close"].rolling(window=window, min_periods=window).mean()

    nan_count = df["rolling_mean"].isna().sum()
    logger.info(
        "Rolling mean computed | NaN rows excluded from signal=%d", nan_count
    )

    valid_mask = df["rolling_mean"].notna()
    df.loc[valid_mask, "signal"] = (
        df.loc[valid_mask, "close"] > df.loc[valid_mask, "rolling_mean"]
    ).astype(int)

    logger.info(
        "Signal generated | valid_rows=%d  signal_1=%d  signal_0=%d",
        valid_mask.sum(),
        int(df.loc[valid_mask, "signal"].sum()),
        int((df.loc[valid_mask, "signal"] == 0).sum()),
    )
    return df


# Metrics output
def write_metrics(metrics: dict, output_path: str, logger: logging.Logger) -> None:
    """Write metrics dict as JSON to output_path and print to stdout."""
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics written to %s", output_path)
    print(json.dumps(metrics, indent=2))



# CLI entry point
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLOps rolling-mean signal pipeline")
    parser.add_argument("--input",    required=True, help="Path to input CSV")
    parser.add_argument("--config",   required=True, help="Path to config YAML")
    parser.add_argument("--output",   required=True, help="Path for output metrics JSON")
    parser.add_argument("--log-file", required=True, help="Path for log file")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = setup_logging(args.log_file)
    start_time = time.time()
    version = "unknown"

    logger.info("=" * 60)
    logger.info("Job started")
    logger.info("  input    : %s", args.input)
    logger.info("  config   : %s", args.config)
    logger.info("  output   : %s", args.output)
    logger.info("  log-file : %s", args.log_file)
    logger.info("=" * 60)

    try:
        # Step 1 — Config
        config = load_config(args.config, logger)
        version = config["version"]

        # Step 2 — Seed
        np.random.seed(config["seed"])
        logger.debug("NumPy random seed set to %d", config["seed"])

        # Step 3 — Data
        df = load_data(args.input, logger)

        # Step 4 — Signals
        df = compute_signals(df, config["window"], logger)

        # Step 5 — Metrics
        valid_signals = df["signal"].dropna()
        rows_processed = len(valid_signals)
        signal_rate = round(float(valid_signals.mean()), 4)
        latency_ms = int((time.time() - start_time) * 1000)

        logger.info(
            "Metrics summary | rows_processed=%d  signal_rate=%.4f  latency_ms=%d",
            rows_processed, signal_rate, latency_ms,
        )

        metrics = {
            "version": version,
            "rows_processed": rows_processed,
            "metric": "signal_rate",
            "value": signal_rate,
            "latency_ms": latency_ms,
            "seed": config["seed"],
            "status": "success",
        }

        write_metrics(metrics, args.output, logger)
        logger.info("Job completed successfully.")
        logger.info("=" * 60)
        return 0

    except Exception as exc:
        latency_ms = int((time.time() - start_time) * 1000)
        logger.error("Job failed: %s", exc, exc_info=True)

        error_metrics = {
            "version": version,
            "status": "error",
            "error_message": str(exc),
            "latency_ms": latency_ms,
        }
        try:
            write_metrics(error_metrics, args.output, logger)
        except Exception as write_exc:
            logger.critical("Could not write error metrics: %s", write_exc)

        logger.info("Job ended with errors.")
        logger.info("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())