"""Configuration and constants for the FreightTiger data pipeline."""

from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
OUTPUT_DATA_DIR = DATA_DIR / "output"

# Raw input file paths
SHIPMENT_RECORDS_FILE = RAW_DATA_DIR / "shipment_records.csv"
CONTEXT_NOTES_FILE = RAW_DATA_DIR / "context_notes.csv"
SAMPLE_OUTPUT_FILE = RAW_DATA_DIR / "sample_output_format_v2.csv"

# Required shipment columns and expected data types
REQUIRED_SHIPMENT_COLUMNS = [
    "shipment_id",
    "origin",
    "destination",
    "route_type",
    "material",
    "quantity_tonnes",
    "distance_km",
    "freight_cost_inr",
    "shipment_date",
    "transporter",
]

VALID_ROUTE_TYPES = {"Short", "Medium", "Long"}

# Baseline calculation settings
HISTORICAL_WINDOW_WEEKS = 8

# Output formatting suffixes
HISTORY_COMPARISON_SUFFIX = "vs this route's past average"
PEER_COMPARISON_SUFFIX = "vs similar-length routes this week"

# Proposed default anomaly detection thresholds (percentage increase >= threshold)
# Inferred from sample output (e.g. Mumbai-Pune peer spike of +23.6%, Ahmedabad-Mumbai +29.5% / +22.5%)
# Note: These are proposed defaults subject to parameterization, not explicitly fixed by FreightTiger.
DEFAULT_HISTORY_THRESHOLD = 20.0
DEFAULT_PEER_THRESHOLD = 20.0

# Output file targets
FINAL_OUTPUT_FILE = OUTPUT_DATA_DIR / "freight_anomaly_report.csv"
RUN_METRICS_FILE = OUTPUT_DATA_DIR / "run_metrics.json"

# LLM and Explainer Configuration
DEFAULT_LLM_PROVIDER = "offline"
DEFAULT_LLM_MODEL = "deterministic-template"

# Pricing rates in USD per 1,000,000 tokens (published standard rates)
MODEL_PRICING_RATES = {
    "offline": {"input_per_m": 0.0, "output_per_m": 0.0},
    "mock": {"input_per_m": 0.15, "output_per_m": 0.60},
    "gpt-4o-mini": {"input_per_m": 0.15, "output_per_m": 0.60},
    "gpt-3.5-turbo": {"input_per_m": 0.50, "output_per_m": 1.50},
    "gemini-1.5-flash": {"input_per_m": 0.075, "output_per_m": 0.30},
    "llama-3.1-8b": {"input_per_m": 0.05, "output_per_m": 0.08},
    "ollama": {"input_per_m": 0.0, "output_per_m": 0.0},
}
