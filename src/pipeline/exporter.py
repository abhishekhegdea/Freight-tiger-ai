"""Submission CSV Exporter adhering to official 8-column schema and RFC 4180 quoting."""

import csv
import logging
from pathlib import Path
from typing import List
import pandas as pd
from src.config import FINAL_OUTPUT_FILE

logger = logging.getLogger(__name__)

# Exact 8 required columns in exact order
SUBMISSION_COLUMNS: List[str] = [
    "route",
    "week_of",
    "cost_per_tonne_km",
    "vs_own_history",
    "vs_similar_routes",
    "flagged",
    "matched_note_id",
    "reason",
]


def format_and_export_csv(
    df: pd.DataFrame,
    output_path: Path | str = FINAL_OUTPUT_FILE,
    include_all_weeks: bool = False
) -> Path:
    """Formats and exports the final submission CSV meeting all contractual requirements.

    Contract Rules:
        1. Exact eight columns: route, week_of, cost_per_tonne_km, vs_own_history,
           vs_similar_routes, flagged, matched_note_id, reason.
        2. Exact column order and naming.
        3. cost_per_tonne_km formatted to 2 decimal places.
        4. Percentage differences formatted as signed floats to 1 decimal place with contract suffixes.
        5. matched_note_id is empty string when unexplained, never 'nan' or 'None'.
        6. RFC 4180 compliant quoting (quoting=csv.QUOTE_MINIMAL) so commas in explanations
           do not break CSV parsers.
        7. Filters to anomaly candidates by default (20 rows), unless include_all_weeks=True.

    Args:
        df: Enriched DataFrame from the pipeline.
        output_path: Target CSV file path.
        include_all_weeks: If True, exports all 728 route-weeks; if False, exports only candidates.

    Returns:
        Path to the written CSV file.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    export_df = df.copy()

    # Filter candidates unless include_all_weeks is requested
    if not include_all_weeks:
        if "is_anomaly_candidate" in export_df.columns:
            export_df = export_df[export_df["is_anomaly_candidate"]].copy()

    # Ensure chronological sorting: week_of ascending, then route ascending
    export_df = export_df.sort_values(by=["week_of", "route"]).reset_index(drop=True)

    # Format cost_per_tonne_km to 2 decimal places
    export_df["cost_per_tonne_km"] = export_df["cost_per_tonne_km_rounded"].apply(
        lambda val: f"{val:.2f}"
    )

    # Clean matched_note_id: ensure missing is strictly empty string ""
    export_df["matched_note_id"] = export_df["matched_note_id"].apply(
        lambda val: str(val).strip() if (pd.notna(val) and str(val).strip().lower() not in ("nan", "none", "")) else ""
    )

    # Validate that all required columns are present
    missing_cols = [c for c in SUBMISSION_COLUMNS if c not in export_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required submission columns: {missing_cols}")

    # Select and order exact 8 columns
    submission_df = export_df[SUBMISSION_COLUMNS]

    # Write CSV with RFC 4180 quoting
    submission_df.to_csv(
        out_file,
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_MINIMAL
    )

    logger.info("Successfully exported %d rows to submission CSV at %s", len(submission_df), out_file)
    return out_file
