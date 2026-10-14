"""Robust shipment CSV loading and data validation module."""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
from src.config import REQUIRED_SHIPMENT_COLUMNS, VALID_ROUTE_TYPES

logger = logging.getLogger(__name__)


class ValidationError(ValueError):
    """Raised when shipment records fail schema, data integrity, or numeric validation."""
    pass


def load_and_validate_shipments(
    file_path: Path | str,
    strict: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Loads shipment records from CSV and applies strict schema and data integrity validations.

    Args:
        file_path: Path to shipment_records.csv.
        strict: If True, raises ValidationError on invalid data. If False, returns report.

    Returns:
        A tuple of (validated_df, validation_report).

    Raises:
        FileNotFoundError: If the input file does not exist.
        ValidationError: If schema or data integrity constraints are violated in strict mode.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Shipment records file not found at: {path}")

    logger.info("Loading shipment records from %s", path)
    df = pd.read_csv(path)

    report: Dict[str, Any] = {
        "total_rows_read": len(df),
        "missing_columns": [],
        "missing_values": {},
        "duplicate_rows": 0,
        "duplicate_shipment_ids": 0,
        "invalid_dates": 0,
        "invalid_quantities": 0,
        "invalid_distances": 0,
        "invalid_freight_costs": 0,
        "invalid_route_types": 0,
        "status": "PASS",
        "errors": []
    }

    # 1. Validate required columns exist
    missing_cols = [col for col in REQUIRED_SHIPMENT_COLUMNS if col not in df.columns]
    if missing_cols:
        report["missing_columns"] = missing_cols
        report["status"] = "FAIL"
        report["errors"].append(f"Missing required columns: {missing_cols}")
        raise ValidationError(f"Schema error: Missing required columns: {missing_cols}")

    # 2. Check for duplicate rows and duplicate shipment_id
    dup_rows = df.duplicated().sum()
    report["duplicate_rows"] = int(dup_rows)
    if dup_rows > 0:
        msg = f"Found {dup_rows} identical duplicate rows."
        report["errors"].append(msg)
        if strict:
            raise ValidationError(msg)

    dup_ids = df["shipment_id"].duplicated().sum()
    report["duplicate_shipment_ids"] = int(dup_ids)
    if dup_ids > 0:
        bad_ids = df[df["shipment_id"].duplicated(keep=False)]["shipment_id"].unique().tolist()
        msg = f"Found duplicate shipment IDs: {bad_ids[:10]}"
        report["errors"].append(msg)
        if strict:
            raise ValidationError(msg)

    # 3. Check for missing (null/NaN) values in any column
    null_counts = df[REQUIRED_SHIPMENT_COLUMNS].isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0].to_dict()
    report["missing_values"] = cols_with_nulls
    if cols_with_nulls:
        msg = f"Missing/null values detected in columns: {cols_with_nulls}"
        report["errors"].append(msg)
        if strict:
            raise ValidationError(msg)

    # 4. Parse dates with error detection
    parsed_dates = pd.to_datetime(df["shipment_date"], errors="coerce")
    invalid_dates_count = int(parsed_dates.isnull().sum())
    report["invalid_dates"] = invalid_dates_count
    if invalid_dates_count > 0:
        msg = f"Found {invalid_dates_count} unparseable shipment dates."
        report["errors"].append(msg)
        if strict:
            raise ValidationError(msg)
    df["shipment_date"] = parsed_dates

    # 5. Validate numeric ranges (quantity > 0, distance > 0, freight_cost > 0)
    invalid_qty = (df["quantity_tonnes"] <= 0) | df["quantity_tonnes"].isna()
    report["invalid_quantities"] = int(invalid_qty.sum())
    if report["invalid_quantities"] > 0:
        msg = f"Found {report['invalid_quantities']} records with non-positive or NaN quantity_tonnes."
        report["errors"].append(msg)
        if strict:
            raise ValidationError(msg)

    invalid_dist = (df["distance_km"] <= 0) | df["distance_km"].isna()
    report["invalid_distances"] = int(invalid_dist.sum())
    if report["invalid_distances"] > 0:
        msg = f"Found {report['invalid_distances']} records with non-positive or NaN distance_km."
        report["errors"].append(msg)
        if strict:
            raise ValidationError(msg)

    invalid_cost = (df["freight_cost_inr"] <= 0) | df["freight_cost_inr"].isna()
    report["invalid_freight_costs"] = int(invalid_cost.sum())
    if report["invalid_freight_costs"] > 0:
        msg = f"Found {report['invalid_freight_costs']} records with non-positive or NaN freight_cost_inr."
        report["errors"].append(msg)
        if strict:
            raise ValidationError(msg)

    # 6. Validate route_type categories
    invalid_rt = ~df["route_type"].isin(VALID_ROUTE_TYPES)
    report["invalid_route_types"] = int(invalid_rt.sum())
    if report["invalid_route_types"] > 0:
        bad_types = df.loc[invalid_rt, "route_type"].unique().tolist()
        msg = f"Found invalid route_type values: {bad_types}. Expected one of {VALID_ROUTE_TYPES}."
        report["errors"].append(msg)
        if strict:
            raise ValidationError(msg)

    # Derive standardized route field: Origin-Destination
    df["route"] = df["origin"].astype(str).str.strip() + "-" + df["destination"].astype(str).str.strip()

    logger.info("Successfully loaded and validated %d shipment records (status: %s)", len(df), report["status"])
    return df, report
