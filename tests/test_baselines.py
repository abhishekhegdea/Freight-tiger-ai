"""Unit tests for baseline calculation: trailing 8-week history and peer baselines."""

import numpy as np
import pandas as pd
import pytest
from src.pipeline.baseline_calculator import compute_baselines_and_deviations


def test_trailing_history_no_look_ahead_and_early_weeks():
    """Verify that history excludes current week and uses only prior weeks."""
    # Create 10 consecutive weeks for RouteA
    weeks = [f"2024-0{i+1:02d}-01" if i < 9 else f"2024-10-01" for i in range(10)]
    # Costs: 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0
    data = []
    for i, w in enumerate(weeks):
        data.append({
            "route": "RouteA",
            "route_type": "Medium",
            "week_of": w,
            "cost_per_tonne_km": float(i + 1),
            "cost_per_tonne_km_rounded": float(i + 1),
            "total_freight_cost_inr": 1000 * (i + 1),
            "total_tonne_km": 1000.0,
            "shipment_count": 3
        })
    df = pd.DataFrame(data)

    enriched = compute_baselines_and_deviations(df, window_weeks=8)

    # Week 0 (Cost 1.0): No prior weeks
    assert pd.isna(enriched.loc[0, "trailing_history_avg"])
    assert enriched.loc[0, "history_prior_weeks_count"] == 0
    assert "N/A" in enriched.loc[0, "vs_own_history"]

    # Week 1 (Cost 2.0): Prior week 0 cost was 1.0
    assert enriched.loc[1, "trailing_history_avg"] == 1.0
    assert enriched.loc[1, "history_prior_weeks_count"] == 1
    # pct change = (2.0 - 1.0) / 1.0 = +100.0%
    assert pytest.approx(enriched.loc[1, "pct_vs_history"], 0.1) == 100.0
    assert "+100.0% vs this route's past average" in enriched.loc[1, "vs_own_history"]

    # Week 8 (Cost 9.0): Prior 8 weeks were costs 1.0 to 8.0 -> mean = 36 / 8 = 4.5
    # Current week (9.0) must NOT be included in its own baseline!
    assert enriched.loc[8, "history_prior_weeks_count"] == 8
    assert pytest.approx(enriched.loc[8, "trailing_history_avg"], 0.001) == 4.5

    # Week 9 (Cost 10.0): Prior 8 weeks were costs 2.0 to 9.0 -> mean = 44 / 8 = 5.5
    assert enriched.loc[9, "history_prior_weeks_count"] == 8
    assert pytest.approx(enriched.loc[9, "trailing_history_avg"], 0.001) == 5.5


def test_peer_baseline_exclusion_and_route_types():
    """Verify that peer calculation excludes the route itself and only compares same route_type."""
    data = [
        # Week 1, Short routes: RouteS1 and RouteS2
        {"route": "RouteS1", "route_type": "Short", "week_of": "2024-01-01", "cost_per_tonne_km": 4.0},
        {"route": "RouteS2", "route_type": "Short", "week_of": "2024-01-01", "cost_per_tonne_km": 2.0},
        # Week 1, Long route: RouteL1 (alone in Long)
        {"route": "RouteL1", "route_type": "Long", "week_of": "2024-01-01", "cost_per_tonne_km": 1.5},
    ]
    df = pd.DataFrame(data)

    enriched = compute_baselines_and_deviations(df)

    s1 = enriched[enriched["route"] == "RouteS1"].iloc[0]
    # S1's peer is S2 only (cost 2.0)
    assert s1["peer_avg"] == 2.0
    assert s1["peer_routes_count"] == 1
    # pct = (4.0 - 2.0) / 2.0 = +100.0%
    assert pytest.approx(s1["pct_vs_peer"], 0.1) == 100.0
    assert "+100.0% vs similar-length routes this week" in s1["vs_similar_routes"]

    s2 = enriched[enriched["route"] == "RouteS2"].iloc[0]
    # S2's peer is S1 only (cost 4.0)
    assert s2["peer_avg"] == 4.0
    assert s2["peer_routes_count"] == 1
    # pct = (2.0 - 4.0) / 4.0 = -50.0%
    assert pytest.approx(s2["pct_vs_peer"], 0.1) == -50.0
    assert "-50.0% vs similar-length routes this week" in s2["vs_similar_routes"]

    l1 = enriched[enriched["route"] == "RouteL1"].iloc[0]
    # L1 has no peers in Long bucket
    assert pd.isna(l1["peer_avg"])
    assert l1["peer_routes_count"] == 0
    assert "N/A" in l1["vs_similar_routes"]


def test_division_by_zero_safety():
    """Verify that zero baselines do not trigger ZeroDivisionError."""
    data = [
        {"route": "RouteZ", "route_type": "Short", "week_of": "2024-01-01", "cost_per_tonne_km": 0.0},
        {"route": "RouteZ", "route_type": "Short", "week_of": "2024-01-08", "cost_per_tonne_km": 2.0},
    ]
    df = pd.DataFrame(data)
    enriched = compute_baselines_and_deviations(df)
    # Week 2 has prior history avg of 0.0
    row_wk2 = enriched[enriched["week_of"] == "2024-01-08"].iloc[0]
    assert pd.isna(row_wk2["pct_vs_history"])
    assert "N/A" in row_wk2["vs_own_history"]
