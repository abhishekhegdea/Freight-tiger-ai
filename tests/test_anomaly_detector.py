"""Unit tests for Phase 2: Deterministic Anomaly Detection Engine."""

import numpy as np
import pandas as pd
import pytest
from src.config import SHIPMENT_RECORDS_FILE
from src.data.loader import load_and_validate_shipments
from src.pipeline.weekly_aggregator import aggregate_weekly_shipments
from src.pipeline.baseline_calculator import compute_baselines_and_deviations
from src.pipeline.anomaly_detector import detect_candidate_anomalies, get_anomaly_summary


def make_test_row(
    route: str = "RouteA",
    week_of: str = "2024-01-08",
    route_type: str = "Medium",
    cost: float = 3.0,
    pct_history: float = 0.0,
    pct_peer: float = 0.0
) -> pd.DataFrame:
    """Helper to generate a single synthetic route-week row."""
    return pd.DataFrame([{
        "route": route,
        "week_of": week_of,
        "route_type": route_type,
        "cost_per_tonne_km": cost,
        "cost_per_tonne_km_rounded": round(cost, 2),
        "pct_vs_history": pct_history,
        "pct_vs_peer": pct_peer,
        "vs_own_history": f"{pct_history:+.1f}% vs history" if pd.notna(pct_history) else "N/A",
        "vs_similar_routes": f"{pct_peer:+.1f}% vs peers" if pd.notna(pct_peer) else "N/A"
    }])


def test_anomaly_historical_exceeded_only():
    """Verify candidate when historical threshold is exceeded but peer is not."""
    df = make_test_row(pct_history=25.0, pct_peer=10.0)
    res = detect_candidate_anomalies(df, history_threshold=20.0, peer_threshold=20.0)
    assert res.iloc[0]["is_history_anomaly"] is True or res.iloc[0]["is_history_anomaly"] == 1
    assert res.iloc[0]["is_peer_anomaly"] is False or res.iloc[0]["is_peer_anomaly"] == 0
    assert res.iloc[0]["is_anomaly_candidate"] is True or res.iloc[0]["is_anomaly_candidate"] == 1
    assert res.iloc[0]["candidate_trigger"] == "HISTORY_ONLY"
    assert "Historical surge" in res.iloc[0]["candidate_reason"]


def test_anomaly_peer_exceeded_only():
    """Verify candidate when peer threshold is exceeded but history is not."""
    df = make_test_row(pct_history=5.0, pct_peer=22.0)
    res = detect_candidate_anomalies(df, history_threshold=20.0, peer_threshold=20.0)
    assert res.iloc[0]["is_history_anomaly"] is False or res.iloc[0]["is_history_anomaly"] == 0
    assert res.iloc[0]["is_peer_anomaly"] is True or res.iloc[0]["is_peer_anomaly"] == 1
    assert res.iloc[0]["is_anomaly_candidate"] is True or res.iloc[0]["is_anomaly_candidate"] == 1
    assert res.iloc[0]["candidate_trigger"] == "PEER_ONLY"
    assert "Peer deviation" in res.iloc[0]["candidate_reason"]


def test_anomaly_both_exceeded():
    """Verify candidate when both thresholds are exceeded."""
    df = make_test_row(pct_history=30.0, pct_peer=25.0)
    res = detect_candidate_anomalies(df, history_threshold=20.0, peer_threshold=20.0)
    assert res.iloc[0]["is_history_anomaly"] is True or res.iloc[0]["is_history_anomaly"] == 1
    assert res.iloc[0]["is_peer_anomaly"] is True or res.iloc[0]["is_peer_anomaly"] == 1
    assert res.iloc[0]["is_anomaly_candidate"] is True or res.iloc[0]["is_anomaly_candidate"] == 1
    assert res.iloc[0]["candidate_trigger"] == "BOTH"
    assert "Historical surge" in res.iloc[0]["candidate_reason"]
    assert "Peer deviation" in res.iloc[0]["candidate_reason"]


def test_anomaly_neither_exceeded():
    """Verify normal status when neither threshold is exceeded."""
    df = make_test_row(pct_history=12.0, pct_peer=8.0)
    res = detect_candidate_anomalies(df, history_threshold=20.0, peer_threshold=20.0)
    assert res.iloc[0]["is_history_anomaly"] is False or res.iloc[0]["is_history_anomaly"] == 0
    assert res.iloc[0]["is_peer_anomaly"] is False or res.iloc[0]["is_peer_anomaly"] == 0
    assert res.iloc[0]["is_anomaly_candidate"] is False or res.iloc[0]["is_anomaly_candidate"] == 0
    assert res.iloc[0]["candidate_trigger"] == "NONE"
    assert "Within normal thresholds" in res.iloc[0]["candidate_reason"]


def test_anomaly_exact_threshold_boundary():
    """Verify that exact boundary value (>= threshold) triggers candidate."""
    # Exactly 20.0% history
    df_hist = make_test_row(pct_history=20.0, pct_peer=0.0)
    res_hist = detect_candidate_anomalies(df_hist, history_threshold=20.0, peer_threshold=20.0)
    assert res_hist.iloc[0]["is_anomaly_candidate"] is True or res_hist.iloc[0]["is_anomaly_candidate"] == 1
    assert res_hist.iloc[0]["is_history_anomaly"] is True or res_hist.iloc[0]["is_history_anomaly"] == 1

    # Exactly 20.0% peer
    df_peer = make_test_row(pct_history=0.0, pct_peer=20.0)
    res_peer = detect_candidate_anomalies(df_peer, history_threshold=20.0, peer_threshold=20.0)
    assert res_peer.iloc[0]["is_anomaly_candidate"] is True or res_peer.iloc[0]["is_anomaly_candidate"] == 1
    assert res_peer.iloc[0]["is_peer_anomaly"] is True or res_peer.iloc[0]["is_peer_anomaly"] == 1


def test_anomaly_negative_percentage_changes():
    """Verify that negative percentage deviations do not trigger anomalies."""
    df = make_test_row(pct_history=-15.0, pct_peer=-25.0)
    res = detect_candidate_anomalies(df, history_threshold=20.0, peer_threshold=20.0)
    assert res.iloc[0]["is_anomaly_candidate"] is False or res.iloc[0]["is_anomaly_candidate"] == 0
    assert res.iloc[0]["candidate_trigger"] == "NONE"


def test_anomaly_missing_historical_baseline():
    """Verify that missing historical baseline (NaN) does not trigger history anomaly."""
    # History missing, peer normal
    df1 = make_test_row(pct_history=np.nan, pct_peer=10.0)
    res1 = detect_candidate_anomalies(df1, history_threshold=20.0, peer_threshold=20.0)
    assert res1.iloc[0]["missing_history_baseline"] is True or res1.iloc[0]["missing_history_baseline"] == 1
    assert res1.iloc[0]["is_history_anomaly"] is False or res1.iloc[0]["is_history_anomaly"] == 0
    assert res1.iloc[0]["is_anomaly_candidate"] is False or res1.iloc[0]["is_anomaly_candidate"] == 0

    # History missing, peer exceeded
    df2 = make_test_row(pct_history=np.nan, pct_peer=25.0)
    res2 = detect_candidate_anomalies(df2, history_threshold=20.0, peer_threshold=20.0)
    assert res2.iloc[0]["missing_history_baseline"] is True or res2.iloc[0]["missing_history_baseline"] == 1
    assert res2.iloc[0]["is_history_anomaly"] is False or res2.iloc[0]["is_history_anomaly"] == 0
    assert res2.iloc[0]["is_peer_anomaly"] is True or res2.iloc[0]["is_peer_anomaly"] == 1
    assert res2.iloc[0]["is_anomaly_candidate"] is True or res2.iloc[0]["is_anomaly_candidate"] == 1
    assert res2.iloc[0]["candidate_trigger"] == "PEER_ONLY"


def test_anomaly_missing_peer_baseline():
    """Verify that missing peer baseline (NaN) does not trigger peer anomaly."""
    # History normal, peer missing
    df1 = make_test_row(pct_history=5.0, pct_peer=np.nan)
    res1 = detect_candidate_anomalies(df1, history_threshold=20.0, peer_threshold=20.0)
    assert res1.iloc[0]["missing_peer_baseline"] is True or res1.iloc[0]["missing_peer_baseline"] == 1
    assert res1.iloc[0]["is_peer_anomaly"] is False or res1.iloc[0]["is_peer_anomaly"] == 0
    assert res1.iloc[0]["is_anomaly_candidate"] is False or res1.iloc[0]["is_anomaly_candidate"] == 0

    # History exceeded, peer missing
    df2 = make_test_row(pct_history=22.0, pct_peer=np.nan)
    res2 = detect_candidate_anomalies(df2, history_threshold=20.0, peer_threshold=20.0)
    assert res2.iloc[0]["missing_peer_baseline"] is True or res2.iloc[0]["missing_peer_baseline"] == 1
    assert res2.iloc[0]["is_peer_anomaly"] is False or res2.iloc[0]["is_peer_anomaly"] == 0
    assert res2.iloc[0]["is_history_anomaly"] is True or res2.iloc[0]["is_history_anomaly"] == 1
    assert res2.iloc[0]["is_anomaly_candidate"] is True or res2.iloc[0]["is_anomaly_candidate"] == 1
    assert res2.iloc[0]["candidate_trigger"] == "HISTORY_ONLY"


def test_anomaly_both_baselines_missing():
    """Verify behavior when both baselines are missing (e.g. unique route in week 1)."""
    df = make_test_row(pct_history=np.nan, pct_peer=np.nan)
    res = detect_candidate_anomalies(df, history_threshold=20.0, peer_threshold=20.0)
    assert res.iloc[0]["missing_history_baseline"] is True or res.iloc[0]["missing_history_baseline"] == 1
    assert res.iloc[0]["missing_peer_baseline"] is True or res.iloc[0]["missing_peer_baseline"] == 1
    assert res.iloc[0]["is_anomaly_candidate"] is False or res.iloc[0]["is_anomaly_candidate"] == 0
    assert res.iloc[0]["candidate_trigger"] == "NONE"


def test_anomaly_custom_thresholds():
    """Verify behavior under custom threshold configurations."""
    df = make_test_row(pct_history=18.0, pct_peer=28.0)

    # Under (15.0, 30.0): history triggers (18 >= 15), peer does not (28 < 30) -> HISTORY_ONLY
    res1 = detect_candidate_anomalies(df, history_threshold=15.0, peer_threshold=30.0)
    assert res1.iloc[0]["candidate_trigger"] == "HISTORY_ONLY"

    # Under (20.0, 25.0): history does not trigger (18 < 20), peer triggers (28 >= 25) -> PEER_ONLY
    res2 = detect_candidate_anomalies(df, history_threshold=20.0, peer_threshold=25.0)
    assert res2.iloc[0]["candidate_trigger"] == "PEER_ONLY"

    # Under (20.0, 30.0): neither triggers -> NONE
    res3 = detect_candidate_anomalies(df, history_threshold=20.0, peer_threshold=30.0)
    assert res3.iloc[0]["candidate_trigger"] == "NONE"


def test_anomaly_repeated_runs_determinism():
    """Verify that multiple consecutive runs on identical data yield bit-for-bit identical results."""
    df, _ = load_and_validate_shipments(SHIPMENT_RECORDS_FILE)
    weekly = aggregate_weekly_shipments(df)
    enriched = compute_baselines_and_deviations(weekly)

    run1 = detect_candidate_anomalies(enriched, history_threshold=20.0, peer_threshold=20.0)
    run2 = detect_candidate_anomalies(enriched, history_threshold=20.0, peer_threshold=20.0)

    pd.testing.assert_frame_equal(run1, run2)


def test_official_sample_rows_are_candidates():
    """Verify that all three official sample rows are identified as candidate anomalies under default thresholds."""
    df, _ = load_and_validate_shipments(SHIPMENT_RECORDS_FILE)
    weekly = aggregate_weekly_shipments(df)
    enriched = compute_baselines_and_deviations(weekly)
    detected = detect_candidate_anomalies(enriched, history_threshold=20.0, peer_threshold=20.0)

    # 1. Ahmedabad-Mumbai (2025-01-20): +29.5% history, +22.5% peers -> Candidate (BOTH)
    row_ahmedabad = detected[
        (detected["route"] == "Ahmedabad-Mumbai") & (detected["week_of"] == "2025-01-20")
    ].iloc[0]
    assert row_ahmedabad["is_anomaly_candidate"] is True or row_ahmedabad["is_anomaly_candidate"] == 1
    assert row_ahmedabad["candidate_trigger"] == "BOTH"

    # 2. Delhi-Jaipur (2024-11-11): +35.5% history, +21.0% peers -> Candidate (BOTH)
    row_delhi = detected[
        (detected["route"] == "Delhi-Jaipur") & (detected["week_of"] == "2024-11-11")
    ].iloc[0]
    assert row_delhi["is_anomaly_candidate"] is True or row_delhi["is_anomaly_candidate"] == 1
    assert row_delhi["candidate_trigger"] == "BOTH"

    # 3. Mumbai-Pune (2025-09-15): +9.2% history, +23.6% peers -> Candidate (PEER_ONLY)
    row_mumbai = detected[
        (detected["route"] == "Mumbai-Pune") & (detected["week_of"] == "2025-09-15")
    ].iloc[0]
    assert row_mumbai["is_anomaly_candidate"] is True or row_mumbai["is_anomaly_candidate"] == 1
    assert row_mumbai["candidate_trigger"] == "PEER_ONLY"
