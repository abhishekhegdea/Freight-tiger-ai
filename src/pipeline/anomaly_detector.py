"""Deterministic Anomaly Detection Engine for FreightTiger Case Study.

Identifies route-week anomaly candidates based on configurable historical and peer thresholds.
Note: Threshold defaults (history: 20.0%, peer: 20.0%) are proposed defaults inferred from
sample case behavior, not explicitly prescribed by FreightTiger specifications.
"""

import logging
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from src.config import DEFAULT_HISTORY_THRESHOLD, DEFAULT_PEER_THRESHOLD

logger = logging.getLogger(__name__)


def detect_candidate_anomalies(
    df: pd.DataFrame,
    history_threshold: float = DEFAULT_HISTORY_THRESHOLD,
    peer_threshold: float = DEFAULT_PEER_THRESHOLD
) -> pd.DataFrame:
    """Evaluates route-week metrics against historical and peer thresholds to identify anomaly candidates.

    A route-week is an anomaly candidate if:
        pct_vs_history >= history_threshold OR pct_vs_peer >= peer_threshold

    Key Safeguards:
        - Missing/undefined baselines (NaN) are NOT treated as 0.0 or negative.
          If a baseline cannot be computed, that comparison is excluded from triggering
          the threshold, and the missing condition is explicitly recorded.
        - Does NOT classify candidates as justified or unexplained (that is deferred to
          Phase 3 RAG & evidence evaluation).
        - Fully deterministic: identical inputs yield identical outputs.

    Args:
        df: DataFrame containing enriched route-week metrics (with pct_vs_history, pct_vs_peer).
        history_threshold: Percentage increase threshold over historical average (default 20.0%).
        peer_threshold: Percentage increase threshold over peer average (default 20.0%).

    Returns:
        DataFrame enriched with anomaly candidate flags, triggers, and diagnostic reasons.
    """
    logger.info(
        "Running anomaly detection on %d records with history_threshold=%.1f%%, peer_threshold=%.1f%%",
        len(df),
        history_threshold,
        peer_threshold
    )

    result = df.copy()

    # Track configured thresholds on every row for traceability
    result["history_threshold"] = float(history_threshold)
    result["peer_threshold"] = float(peer_threshold)

    # 1. Identify missing baselines explicitly
    result["missing_history_baseline"] = result["pct_vs_history"].isna()
    result["missing_peer_baseline"] = result["pct_vs_peer"].isna()

    # 2. Evaluate thresholds (excluding missing baselines; never treating NaN as zero)
    result["is_history_anomaly"] = (
        (~result["missing_history_baseline"]) & (result["pct_vs_history"] >= history_threshold)
    )
    result["is_peer_anomaly"] = (
        (~result["missing_peer_baseline"]) & (result["pct_vs_peer"] >= peer_threshold)
    )

    # 3. Candidate decision: OR logic
    result["is_anomaly_candidate"] = result["is_history_anomaly"] | result["is_peer_anomaly"]

    # 4. Trigger categorization
    def categorize_trigger(row: pd.Series) -> str:
        if row["is_history_anomaly"] and row["is_peer_anomaly"]:
            return "BOTH"
        elif row["is_history_anomaly"]:
            return "HISTORY_ONLY"
        elif row["is_peer_anomaly"]:
            return "PEER_ONLY"
        else:
            return "NONE"

    result["candidate_trigger"] = result.apply(categorize_trigger, axis=1)

    # 5. Diagnostic trigger reason (deterministic explanation of qualification)
    def build_trigger_reason(row: pd.Series) -> str:
        reasons: List[str] = []
        if row["is_history_anomaly"]:
            reasons.append(
                f"Historical surge ({row['pct_vs_history']:+.1f}% >= {history_threshold:.1f}%)"
            )
        elif row["missing_history_baseline"]:
            reasons.append("Historical baseline undefined (insufficient history)")

        if row["is_peer_anomaly"]:
            reasons.append(
                f"Peer deviation ({row['pct_vs_peer']:+.1f}% >= {peer_threshold:.1f}%)"
            )
        elif row["missing_peer_baseline"]:
            reasons.append("Peer baseline undefined (no peer routes in group)")

        if not row["is_anomaly_candidate"]:
            if not reasons:
                return "Within normal thresholds"
            return f"Within normal thresholds; {'; '.join(reasons)}"

        return "; ".join(reasons)

    result["candidate_reason"] = result.apply(build_trigger_reason, axis=1)

    candidate_count = int(result["is_anomaly_candidate"].sum())
    logger.info("Identified %d anomaly candidates out of %d records", candidate_count, len(result))

    return result


def get_anomaly_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Generates diagnostic summary statistics for anomaly detection results."""
    candidates = df[df["is_anomaly_candidate"]]

    summary = {
        "total_records": len(df),
        "total_candidates": int(len(candidates)),
        "candidate_rate_pct": round((len(candidates) / len(df) * 100), 2) if len(df) > 0 else 0.0,
        "history_threshold": float(df["history_threshold"].iloc[0]) if "history_threshold" in df.columns else DEFAULT_HISTORY_THRESHOLD,
        "peer_threshold": float(df["peer_threshold"].iloc[0]) if "peer_threshold" in df.columns else DEFAULT_PEER_THRESHOLD,
        "by_route_type": candidates["route_type"].value_counts().to_dict() if "route_type" in candidates.columns else {},
        "by_trigger": candidates["candidate_trigger"].value_counts().to_dict(),
        "missing_baselines": {
            "missing_history_only": int((df["missing_history_baseline"] & ~df["missing_peer_baseline"]).sum()),
            "missing_peer_only": int((~df["missing_history_baseline"] & df["missing_peer_baseline"]).sum()),
            "missing_both": int((df["missing_history_baseline"] & df["missing_peer_baseline"]).sum()),
        }
    }
    return summary
