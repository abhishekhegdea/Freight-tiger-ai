"""Verification of official sample output ground-truth rows."""

import pytest
from src.config import SHIPMENT_RECORDS_FILE
from src.data.loader import load_and_validate_shipments
from src.pipeline.weekly_aggregator import aggregate_weekly_shipments
from src.pipeline.baseline_calculator import compute_baselines_and_deviations


@pytest.fixture(scope="module")
def processed_shipment_data():
    """Fixture that loads, validates, aggregates, and computes baselines for real shipment records."""
    df, report = load_and_validate_shipments(SHIPMENT_RECORDS_FILE, strict=True)
    assert report["status"] == "PASS"
    weekly = aggregate_weekly_shipments(df)
    enriched = compute_baselines_and_deviations(weekly)
    return enriched


def test_sample_row_ahmedabad_mumbai(processed_shipment_data):
    """Verify Ahmedabad-Mumbai on 2025-01-20 matches official contract values."""
    row = processed_shipment_data[
        (processed_shipment_data["route"] == "Ahmedabad-Mumbai") &
        (processed_shipment_data["week_of"] == "2025-01-20")
    ].iloc[0]

    # Expected: cost 3.29, history +29.5%, peers +22.5%
    assert pytest.approx(row["cost_per_tonne_km_rounded"], 0.01) == 3.29
    assert pytest.approx(row["pct_vs_history"], 0.1) == 29.5
    assert pytest.approx(row["pct_vs_peer"], 0.1) == 22.5
    assert row["vs_own_history"] == "+29.5% vs this route's past average"
    assert row["vs_similar_routes"] == "+22.5% vs similar-length routes this week"


def test_sample_row_delhi_jaipur(processed_shipment_data):
    """Verify Delhi-Jaipur on 2024-11-11 matches official contract values."""
    row = processed_shipment_data[
        (processed_shipment_data["route"] == "Delhi-Jaipur") &
        (processed_shipment_data["week_of"] == "2024-11-11")
    ].iloc[0]

    # Expected: cost 4.17, history +35.5%, peers +21.0%
    assert pytest.approx(row["cost_per_tonne_km_rounded"], 0.01) == 4.17
    assert pytest.approx(row["pct_vs_history"], 0.1) == 35.5
    assert pytest.approx(row["pct_vs_peer"], 0.1) == 21.0
    assert row["vs_own_history"] == "+35.5% vs this route's past average"
    assert row["vs_similar_routes"] == "+21.0% vs similar-length routes this week"


def test_sample_row_mumbai_pune(processed_shipment_data):
    """Verify Mumbai-Pune on 2025-09-15 matches official contract values."""
    row = processed_shipment_data[
        (processed_shipment_data["route"] == "Mumbai-Pune") &
        (processed_shipment_data["week_of"] == "2025-09-15")
    ].iloc[0]

    # Expected: cost 3.98, history +9.2%, peers +23.6%
    assert pytest.approx(row["cost_per_tonne_km_rounded"], 0.01) == 3.98
    assert pytest.approx(row["pct_vs_history"], 0.1) == 9.2
    assert pytest.approx(row["pct_vs_peer"], 0.1) == 23.6
    assert row["vs_own_history"] == "+9.2% vs this route's past average"
    assert row["vs_similar_routes"] == "+23.6% vs similar-length routes this week"
