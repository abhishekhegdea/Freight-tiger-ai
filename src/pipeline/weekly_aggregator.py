"""Weekly shipment aggregation and cost per tonne-km computation."""

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def compute_monday_week_of(dates: pd.Series) -> pd.Series:
    """Computes the Monday date string (YYYY-MM-DD) for each date.

    Weeks run Monday to Sunday; week_of is strictly that week's Monday date.
    Formula: date - Timedelta(days=date.weekday).
    """
    dt_series = pd.to_datetime(dates)
    monday_series = dt_series.apply(lambda d: (d - pd.Timedelta(days=d.weekday())).strftime("%Y-%m-%d"))
    return monday_series


def aggregate_weekly_shipments(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates individual shipment records by route, route_type, and week_of (Monday date).

    Calculates:
        tonne_km = quantity_tonnes * distance_km
        cost_per_tonne_km = sum(freight_cost_inr) / sum(tonne_km)

    Internal precision is preserved in `cost_per_tonne_km` (float64).
    Presentation rounding (2 decimal places) is provided in `cost_per_tonne_km_rounded`.

    Args:
        df: Validated shipments DataFrame containing required columns.

    Returns:
        Aggregated DataFrame sorted chronologically by route and week_of.
    """
    logger.info("Aggregating %d shipment records into route-week groups", len(df))

    df_copy = df.copy()

    # Ensure route is formatted as Origin-Destination
    if "route" not in df_copy.columns:
        df_copy["route"] = df_copy["origin"].astype(str).str.strip() + "-" + df_copy["destination"].astype(str).str.strip()

    # Compute Monday week_of
    df_copy["week_of"] = compute_monday_week_of(df_copy["shipment_date"])

    # Compute individual shipment tonne-km
    df_copy["tonne_km"] = df_copy["quantity_tonnes"] * df_copy["distance_km"]

    # Group by route, route_type, and week_of
    grouped = df_copy.groupby(["route", "route_type", "week_of"], as_index=False).agg(
        total_freight_cost_inr=("freight_cost_inr", "sum"),
        total_tonne_km=("tonne_km", "sum"),
        total_quantity_tonnes=("quantity_tonnes", "sum"),
        shipment_count=("shipment_id", "count")
    )

    # Calculate cost per tonne-km: sum(freight_cost_inr) / sum(tonne_km)
    # Guard against division by zero if total_tonne_km is zero
    grouped["cost_per_tonne_km"] = grouped.apply(
        lambda row: (row["total_freight_cost_inr"] / row["total_tonne_km"]) if row["total_tonne_km"] > 0 else 0.0,
        axis=1
    )

    # Presentation rounding: 2 decimal places
    grouped["cost_per_tonne_km_rounded"] = grouped["cost_per_tonne_km"].round(2)

    # Sort deterministically by route and week_of
    grouped = grouped.sort_values(by=["route", "week_of"]).reset_index(drop=True)

    logger.info("Generated %d route-week records across %d unique routes", len(grouped), grouped["route"].nunique())
    return grouped
