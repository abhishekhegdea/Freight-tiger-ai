"""Unit tests for weekly aggregation and cost per tonne-km computation."""

import pandas as pd
import pytest
from src.pipeline.weekly_aggregator import compute_monday_week_of, aggregate_weekly_shipments


def test_compute_monday_week_of_boundaries():
    """Verify that dates from Monday to Sunday map to that week's Monday date."""
    dates = pd.Series([
        "2024-01-01",  # Monday -> 2024-01-01
        "2024-01-02",  # Tuesday -> 2024-01-01
        "2024-01-05",  # Friday -> 2024-01-01
        "2024-01-06",  # Saturday -> 2024-01-01
        "2024-01-07",  # Sunday -> 2024-01-01
        "2024-01-08",  # Next Monday -> 2024-01-08
    ])
    mondays = compute_monday_week_of(dates)
    expected = [
        "2024-01-01",
        "2024-01-01",
        "2024-01-01",
        "2024-01-01",
        "2024-01-01",
        "2024-01-08",
    ]
    assert mondays.tolist() == expected


def test_aggregate_weekly_shipments_formula():
    """Verify cost_per_tonne_km = sum(cost) / sum(quantity * distance)."""
    # Create 2 shipments in the same route and same week
    # Shipment 1: 10 tonnes, 100 km, cost 2000 -> tonne_km = 1000
    # Shipment 2: 20 tonnes, 200 km, cost 6000 -> tonne_km = 4000
    # Total cost = 8000, Total tonne_km = 5000 -> cost_per_tonne_km = 8000 / 5000 = 1.6
    sample_data = pd.DataFrame([
        {
            "shipment_id": "S1",
            "origin": "CityA",
            "destination": "CityB",
            "route_type": "Short",
            "material": "Cement",
            "quantity_tonnes": 10.0,
            "distance_km": 100.0,
            "freight_cost_inr": 2000,
            "shipment_date": "2024-01-01",
            "transporter": "Carrier1"
        },
        {
            "shipment_id": "S2",
            "origin": "CityA",
            "destination": "CityB",
            "route_type": "Short",
            "material": "Cement",
            "quantity_tonnes": 20.0,
            "distance_km": 200.0,
            "freight_cost_inr": 6000,
            "shipment_date": "2024-01-03",
            "transporter": "Carrier2"
        }
    ])

    weekly = aggregate_weekly_shipments(sample_data)
    assert len(weekly) == 1
    row = weekly.iloc[0]
    assert row["route"] == "CityA-CityB"
    assert row["week_of"] == "2024-01-01"
    assert row["total_freight_cost_inr"] == 8000
    assert row["total_tonne_km"] == 5000.0
    assert pytest.approx(row["cost_per_tonne_km"], 0.0001) == 1.6
    assert row["cost_per_tonne_km_rounded"] == 1.60
    assert row["shipment_count"] == 2


def test_aggregate_weekly_division_by_zero_protection():
    """Verify that if tonne_km is zero, calculation does not crash and yields 0.0."""
    sample_data = pd.DataFrame([
        {
            "shipment_id": "S1",
            "origin": "CityA",
            "destination": "CityB",
            "route_type": "Short",
            "material": "Cement",
            "quantity_tonnes": 0.0,
            "distance_km": 0.0,
            "freight_cost_inr": 0,
            "shipment_date": "2024-01-01",
            "transporter": "Carrier1"
        }
    ])
    weekly = aggregate_weekly_shipments(sample_data)
    assert len(weekly) == 1
    assert weekly.iloc[0]["cost_per_tonne_km"] == 0.0
