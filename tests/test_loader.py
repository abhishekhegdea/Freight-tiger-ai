"""Unit tests for CSV loading and validation logic."""

import tempfile
from pathlib import Path
import pandas as pd
import pytest
from src.data.loader import load_and_validate_shipments, ValidationError


def create_temp_csv(rows: list, columns: list = None) -> Path:
    """Helper to create a temporary CSV file."""
    f = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv")
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(f.name, index=False)
    f.close()
    return Path(f.name)


def test_loader_missing_columns():
    """Verify that missing required columns raises ValidationError."""
    csv_path = create_temp_csv([{"shipment_id": "S1", "origin": "A"}])
    with pytest.raises(ValidationError, match="Missing required columns"):
        load_and_validate_shipments(csv_path, strict=True)


def test_loader_non_positive_quantity():
    """Verify that zero or negative quantity_tonnes raises ValidationError."""
    valid_row = {
        "shipment_id": "S1",
        "origin": "Mumbai",
        "destination": "Pune",
        "route_type": "Short",
        "material": "Cement",
        "quantity_tonnes": 0.0,  # Invalid
        "distance_km": 150.0,
        "freight_cost_inr": 5000,
        "shipment_date": "2024-01-01",
        "transporter": "TransCo"
    }
    csv_path = create_temp_csv([valid_row])
    with pytest.raises(ValidationError, match="quantity_tonnes"):
        load_and_validate_shipments(csv_path, strict=True)


def test_loader_duplicate_rows():
    """Verify that identical duplicate rows are detected and rejected."""
    row1 = {
        "shipment_id": "S1",
        "origin": "Mumbai",
        "destination": "Pune",
        "route_type": "Short",
        "material": "Cement",
        "quantity_tonnes": 10.0,
        "distance_km": 150.0,
        "freight_cost_inr": 5000,
        "shipment_date": "2024-01-01",
        "transporter": "TransCo"
    }
    row2 = dict(row1)  # Exact identical row
    csv_path = create_temp_csv([row1, row2])
    with pytest.raises(ValidationError, match="identical duplicate rows"):
        load_and_validate_shipments(csv_path, strict=True)


def test_loader_duplicate_shipment_id():
    """Verify that duplicate shipment_ids with different row values are detected and rejected."""
    row1 = {
        "shipment_id": "S1",
        "origin": "Mumbai",
        "destination": "Pune",
        "route_type": "Short",
        "material": "Cement",
        "quantity_tonnes": 10.0,
        "distance_km": 150.0,
        "freight_cost_inr": 5000,
        "shipment_date": "2024-01-01",
        "transporter": "TransCo"
    }
    row2 = dict(row1)
    row2["freight_cost_inr"] = 6000  # Different cost, same shipment_id S1
    csv_path = create_temp_csv([row1, row2])
    with pytest.raises(ValidationError, match="duplicate shipment IDs"):
        load_and_validate_shipments(csv_path, strict=True)


def test_loader_invalid_route_type():
    """Verify that unrecognized route_type raises ValidationError."""
    row = {
        "shipment_id": "S1",
        "origin": "Mumbai",
        "destination": "Pune",
        "route_type": "UltraLong",  # Invalid
        "material": "Cement",
        "quantity_tonnes": 10.0,
        "distance_km": 150.0,
        "freight_cost_inr": 5000,
        "shipment_date": "2024-01-01",
        "transporter": "TransCo"
    }
    csv_path = create_temp_csv([row])
    with pytest.raises(ValidationError, match="invalid route_type"):
        load_and_validate_shipments(csv_path, strict=True)
