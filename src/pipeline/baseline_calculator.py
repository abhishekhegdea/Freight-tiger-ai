"""Baseline metrics computation module: trailing 8-week history and same-route-type peer comparisons."""

import logging
import numpy as np
import pandas as pd
from src.config import (
    HISTORICAL_WINDOW_WEEKS,
    HISTORY_COMPARISON_SUFFIX,
    PEER_COMPARISON_SUFFIX,
)

logger = logging.getLogger(__name__)


def compute_baselines_and_deviations(
    weekly_df: pd.DataFrame,
    window_weeks: int = HISTORICAL_WINDOW_WEEKS
) -> pd.DataFrame:
    """Computes trailing historical baselines, peer group baselines, and percentage deviations.

    Historical Baseline (vs. own history):
        - Rolling average of cost_per_tonne_km for up to `window_weeks` strictly before the current week.
        - Strict no look-ahead: current week and future weeks are excluded (shift(1)).
        - Early weeks: if fewer than window_weeks exist, uses all available prior weeks (min_periods=1).
        - Week 1 has no prior history: baseline is np.nan.

    Peer Baseline (vs. similar routes):
        - Average cost_per_tonne_km in the same week across all OTHER routes sharing the same route_type.
        - The route itself is strictly excluded from its own peer average.
        - If no peers exist, baseline is np.nan.

    Formatting:
        - Deviations are formatted as:
            '+35.5% vs this route\'s past average'
            '+21.0% vs similar-length routes this week'
        - Missing baselines are clearly indicated (e.g. 'N/A (no prior history)').

    Args:
        weekly_df: Aggregated route-week DataFrame from weekly_aggregator.
        window_weeks: Number of historical trailing weeks (default 8).

    Returns:
        DataFrame enriched with baseline metrics and formatted comparison columns.
    """
    logger.info("Computing baselines for %d route-week records (historical window=%d weeks)", len(weekly_df), window_weeks)

    df = weekly_df.copy()

    # Ensure chronological ordering per route
    df = df.sort_values(by=["route", "week_of"]).reset_index(drop=True)

    # Initialize baseline columns
    df["trailing_history_avg"] = np.nan
    df["history_prior_weeks_count"] = 0
    df["peer_avg"] = np.nan
    df["peer_routes_count"] = 0
    df["pct_vs_history"] = np.nan
    df["pct_vs_peer"] = np.nan
    df["vs_own_history"] = ""
    df["vs_similar_routes"] = ""

    # 1. Compute Historical Baseline (per route, strictly prior weeks)
    for route in df["route"].unique():
        r_indices = df[df["route"] == route].index
        costs = df.loc[r_indices, "cost_per_tonne_km"]

        # shift(1) guarantees strictly prior weeks (no look-ahead)
        prior_costs = costs.shift(1)
        rolling_mean = prior_costs.rolling(window=window_weeks, min_periods=1).mean()
        rolling_count = prior_costs.rolling(window=window_weeks, min_periods=1).count()

        df.loc[r_indices, "trailing_history_avg"] = rolling_mean
        df.loc[r_indices, "history_prior_weeks_count"] = rolling_count.fillna(0).astype(int)

    # 2. Compute Peer Baseline (same week_of, same route_type, route != current_route)
    for idx, row in df.iterrows():
        peers = df[
            (df["week_of"] == row["week_of"]) &
            (df["route_type"] == row["route_type"]) &
            (df["route"] != row["route"])
        ]
        if len(peers) > 0:
            peer_mean = peers["cost_per_tonne_km"].mean()
            df.loc[idx, "peer_avg"] = peer_mean
            df.loc[idx, "peer_routes_count"] = len(peers)

    # 3. Calculate percentage deviations and format strings
    # History deviation: ((current - history) / history) * 100
    valid_hist_mask = df["trailing_history_avg"].notna() & (df["trailing_history_avg"] > 0)
    df.loc[valid_hist_mask, "pct_vs_history"] = (
        (df.loc[valid_hist_mask, "cost_per_tonne_km"] - df.loc[valid_hist_mask, "trailing_history_avg"])
        / df.loc[valid_hist_mask, "trailing_history_avg"]
    ) * 100

    df["vs_own_history"] = df["pct_vs_history"].apply(
        lambda val: f"{val:+.1f}% {HISTORY_COMPARISON_SUFFIX}" if pd.notna(val) else f"N/A (no prior history)"
    )

    # Peer deviation: ((current - peer) / peer) * 100
    valid_peer_mask = df["peer_avg"].notna() & (df["peer_avg"] > 0)
    df.loc[valid_peer_mask, "pct_vs_peer"] = (
        (df.loc[valid_peer_mask, "cost_per_tonne_km"] - df.loc[valid_peer_mask, "peer_avg"])
        / df.loc[valid_peer_mask, "peer_avg"]
    ) * 100

    df["vs_similar_routes"] = df["pct_vs_peer"].apply(
        lambda val: f"{val:+.1f}% {PEER_COMPARISON_SUFFIX}" if pd.notna(val) else f"N/A (no peer routes)"
    )

    logger.info("Successfully computed historical and peer comparisons")
    return df
