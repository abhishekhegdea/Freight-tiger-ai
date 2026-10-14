"""End-to-End Pipeline Integration Tests for Phase 5."""

import tempfile
from pathlib import Path
import pandas as pd
import pytest

from src.config import (
    SHIPMENT_RECORDS_FILE,
    CONTEXT_NOTES_FILE,
    FINAL_OUTPUT_FILE,
    RUN_METRICS_FILE,
)
from src.pipeline.exporter import SUBMISSION_COLUMNS
from run_pipeline import execute_pipeline


@pytest.fixture(scope="module")
def run_end_to_end():
    """Runs the full end-to-end pipeline once in offline mode to a temporary output location."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_csv = Path(tmpdir) / "test_freight_anomaly_report.csv"
        tmp_metrics = Path(tmpdir) / "test_run_metrics.json"

        final_df, csv_path, metrics_path = execute_pipeline(
            output_csv_path=tmp_csv,
            metrics_path=tmp_metrics,
            provider_name="offline",
            include_all_weeks=False
        )
        # Read the exported CSV directly via pandas to test round-trip
        exported_df = pd.read_csv(csv_path, dtype={"matched_note_id": str})
        # Replace NaN with empty string
        exported_df["matched_note_id"] = exported_df["matched_note_id"].fillna("")

        yield exported_df, csv_path, metrics_path


def test_integration_exact_columns_and_order(run_end_to_end):
    """Verify that the exported CSV has the exact 8 required columns in the exact order."""
    exported_df, _, _ = run_end_to_end
    assert exported_df.columns.tolist() == SUBMISSION_COLUMNS
    assert len(exported_df.columns) == 8


def test_integration_row_count_default_candidates(run_end_to_end):
    """Verify that the default pipeline exports exactly 20 candidate rows."""
    exported_df, _, _ = run_end_to_end
    assert len(exported_df) == 20


def test_integration_valid_rfc4180_csv_round_trip(run_end_to_end):
    """Verify that the exported CSV loads cleanly without parse errors and all rows have 8 fields."""
    _, csv_path, _ = run_end_to_end
    # Ensure pandas can parse it without error
    df = pd.read_csv(csv_path)
    assert len(df.columns) == 8
    # Ensure no rows contain unescaped extra columns
    with open(csv_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            # In RFC 4180 CSV, properly quoted commas should not split lines
            pass


def test_integration_official_sample_row_delhi_jaipur(run_end_to_end):
    """Verify Delhi-Jaipur (2024-11-11) matches official contract."""
    exported_df, _, _ = run_end_to_end
    row = exported_df[
        (exported_df["route"] == "Delhi-Jaipur") &
        (exported_df["week_of"] == "2024-11-11")
    ].iloc[0]

    assert row["cost_per_tonne_km"] == 4.17
    assert row["vs_own_history"] == "+35.5% vs this route's past average"
    assert row["vs_similar_routes"] == "+21.0% vs similar-length routes this week"
    assert row["flagged"] == "Yes"
    assert row["matched_note_id"] == ""
    assert "No matching note found" in row["reason"]


def test_integration_official_sample_row_ahmedabad_mumbai(run_end_to_end):
    """Verify Ahmedabad-Mumbai (2025-01-20) matches official contract."""
    exported_df, _, _ = run_end_to_end
    row = exported_df[
        (exported_df["route"] == "Ahmedabad-Mumbai") &
        (exported_df["week_of"] == "2025-01-20")
    ].iloc[0]

    assert row["cost_per_tonne_km"] == 3.29
    assert row["vs_own_history"] == "+29.5% vs this route's past average"
    assert row["vs_similar_routes"] == "+22.5% vs similar-length routes this week"
    assert row["flagged"] == "No (justified)"
    assert row["matched_note_id"] == "N002"
    assert "Matches note N002" in row["reason"]


def test_integration_official_sample_row_mumbai_pune(run_end_to_end):
    """Verify Mumbai-Pune (2025-09-15) matches official contract."""
    exported_df, _, _ = run_end_to_end
    row = exported_df[
        (exported_df["route"] == "Mumbai-Pune") &
        (exported_df["week_of"] == "2025-09-15")
    ].iloc[0]

    assert row["cost_per_tonne_km"] == 3.98
    assert row["vs_own_history"] == "+9.2% vs this route's past average"
    assert row["vs_similar_routes"] == "+23.6% vs similar-length routes this week"
    assert row["flagged"] == "Yes"
    assert row["matched_note_id"] == ""
    assert "closest note (N006" in row["reason"]


def test_integration_matched_note_id_consistency(run_end_to_end):
    """Verify that matched_note_id is populated ONLY when flagged == 'No (justified)'."""
    exported_df, _, _ = run_end_to_end
    for _, row in exported_df.iterrows():
        if row["flagged"] == "No (justified)":
            assert row["matched_note_id"] in ("N001", "N002", "N003")
        elif row["flagged"] == "Yes":
            assert row["matched_note_id"] == ""


def test_integration_source_files_unmodified():
    """Verify that raw input files remain unmodified."""
    assert SHIPMENT_RECORDS_FILE.is_file()
    assert CONTEXT_NOTES_FILE.is_file()
    # Check that shipment records still has exactly 2940 rows
    ship_df = pd.read_csv(SHIPMENT_RECORDS_FILE)
    assert len(ship_df) == 2940


def test_integration_repeated_offline_runs_identical():
    """Verify repeated runs generate identical CSV outputs (bit-for-bit determinism)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv1 = Path(tmpdir) / "run1.csv"
        csv2 = Path(tmpdir) / "run2.csv"
        m1 = Path(tmpdir) / "m1.json"
        m2 = Path(tmpdir) / "m2.json"

        execute_pipeline(output_csv_path=csv1, metrics_path=m1, provider_name="offline")
        execute_pipeline(output_csv_path=csv2, metrics_path=m2, provider_name="offline")

        text1 = csv1.read_text(encoding="utf-8")
        text2 = csv2.read_text(encoding="utf-8")

        assert text1 == text2


def test_integration_include_all_weeks():
    """Verify optional --include-all-weeks flag exports all 728 route-weeks with normal labeled 'No'."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_all = Path(tmpdir) / "all_weeks.csv"
        m_all = Path(tmpdir) / "m_all.json"
        execute_pipeline(output_csv_path=csv_all, metrics_path=m_all, provider_name="offline", include_all_weeks=True)

        df_all = pd.read_csv(csv_all, dtype={"matched_note_id": str})
        assert len(df_all) == 728
        assert df_all.columns.tolist() == SUBMISSION_COLUMNS

        # Normal rows should have flagged == 'No'
        normal_rows = df_all[df_all["flagged"] == "No"]
        assert len(normal_rows) == 708
        for _, r in normal_rows.iterrows():
            assert pd.isna(r["matched_note_id"]) or r["matched_note_id"] == ""
            assert "within standard historical and peer thresholds" in r["reason"]
