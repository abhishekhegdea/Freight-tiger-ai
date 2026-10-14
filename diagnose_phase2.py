"""Diagnostic CLI script for Phase 2: Anomaly Detection Engine Verification."""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    SHIPMENT_RECORDS_FILE,
    DEFAULT_HISTORY_THRESHOLD,
    DEFAULT_PEER_THRESHOLD
)
from src.data.loader import load_and_validate_shipments
from src.pipeline.weekly_aggregator import aggregate_weekly_shipments
from src.pipeline.baseline_calculator import compute_baselines_and_deviations
from src.pipeline.anomaly_detector import detect_candidate_anomalies, get_anomaly_summary


def run_phase2_diagnostics(
    file_path: Path = SHIPMENT_RECORDS_FILE,
    history_threshold: float = DEFAULT_HISTORY_THRESHOLD,
    peer_threshold: float = DEFAULT_PEER_THRESHOLD
) -> pd.DataFrame:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 80)
    print("FREIGHTTIGER PHASE 2: ANOMALY DETECTION ENGINE DIAGNOSTICS")
    print(f"Configured Thresholds: History >= {history_threshold:.1f}% | Peer >= {peer_threshold:.1f}%")
    print("(Note: Inferred defaults from sample cases; not fixed by FreightTiger specification)")
    print("=" * 80)

    # 1. Pipeline execution: Load -> Aggregate -> Baselines -> Detect Candidates
    df, report = load_and_validate_shipments(file_path, strict=True)
    weekly = aggregate_weekly_shipments(df)
    enriched = compute_baselines_and_deviations(weekly)
    detected = detect_candidate_anomalies(
        enriched,
        history_threshold=history_threshold,
        peer_threshold=peer_threshold
    )

    # 2. Extract summary metrics
    summary = get_anomaly_summary(detected)
    candidates = detected[detected["is_anomaly_candidate"]].copy()

    print("\n--- Summary Statistics ---")
    print(f"Total route-week pairs processed : {summary['total_records']}")
    print(f"Total candidate anomalies        : {summary['total_candidates']} ({summary['candidate_rate_pct']}%)")
    print(f"Non-candidate (normal) records   : {summary['total_records'] - summary['total_candidates']}")

    print("\n--- Candidate Counts by Route Type ---")
    for rtype, count in summary["by_route_type"].items():
        print(f"  - {rtype:<8}: {count:>3} candidates")

    print("\n--- Candidate Counts by Trigger Category ---")
    trigger_counts = summary["by_trigger"]
    print(f"  - BOTH History & Peer exceeded : {trigger_counts.get('BOTH', 0):>3}")
    print(f"  - HISTORY_ONLY exceeded        : {trigger_counts.get('HISTORY_ONLY', 0):>3}")
    print(f"  - PEER_ONLY exceeded           : {trigger_counts.get('PEER_ONLY', 0):>3}")

    print("\n--- Missing Baseline Tracking (Diagnostics) ---")
    missing = summary["missing_baselines"]
    print(f"  - Missing history baseline only (Week 1) : {missing['missing_history_only']}")
    print(f"  - Missing peer baseline only             : {missing['missing_peer_only']}")
    print(f"  - Missing both baselines                 : {missing['missing_both']}")

    # 3. Check official sample rows
    print("\n--- Official Sample Cases Regression Check ---")
    sample_cases = [
        ("Ahmedabad-Mumbai", "2025-01-20"),
        ("Delhi-Jaipur", "2024-11-11"),
        ("Mumbai-Pune", "2025-09-15"),
    ]
    for route, week_of in sample_cases:
        match = detected[(detected["route"] == route) & (detected["week_of"] == week_of)]
        if not match.empty:
            row = match.iloc[0]
            status = "[CANDIDATE]" if row["is_anomaly_candidate"] else "[NOT CANDIDATE]"
            print(f"  {route:<18} ({week_of}): {status:<15} Trigger={row['candidate_trigger']:<12} | "
                  f"Cost={row['cost_per_tonne_km_rounded']:.2f} | Hist={row['vs_own_history']} | Peer={row['vs_similar_routes']}")

    # 4. Table of candidate route-weeks
    print("\n--- Shortlist of All Candidate Route-Weeks (Numeric Comparisons) ---")
    candidates_display = candidates[[
        "route", "route_type", "week_of", "cost_per_tonne_km_rounded",
        "pct_vs_history", "pct_vs_peer", "candidate_trigger"
    ]].sort_values(by=["week_of", "route"]).reset_index(drop=True)

    header = f"{'Route':<20} {'Type':<8} {'Week Of':<12} {'Cost':<6} {'Hist %':<10} {'Peer %':<10} {'Trigger':<12}"
    print(header)
    print("-" * len(header))
    for _, r in candidates_display.iterrows():
        hist_str = f"{r['pct_vs_history']:+.1f}%" if pd.notna(r['pct_vs_history']) else "N/A"
        peer_str = f"{r['pct_vs_peer']:+.1f}%" if pd.notna(r['pct_vs_peer']) else "N/A"
        print(f"{r['route']:<20} {r['route_type']:<8} {r['week_of']:<12} {r['cost_per_tonne_km_rounded']:<6.2f} "
              f"{hist_str:<10} {peer_str:<10} {r['candidate_trigger']:<12}")

    print("\n" + "=" * 80)
    print("PHASE 2 DIAGNOSTICS COMPLETE")
    print("=" * 80)

    return detected


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FreightTiger Phase 2: Anomaly Detection Engine Diagnostics")
    parser.add_argument(
        "--history-threshold",
        type=float,
        default=DEFAULT_HISTORY_THRESHOLD,
        help="Percentage increase over historical average to trigger candidate anomaly (default: 20.0)"
    )
    parser.add_argument(
        "--peer-threshold",
        type=float,
        default=DEFAULT_PEER_THRESHOLD,
        help="Percentage increase over peer average to trigger candidate anomaly (default: 20.0)"
    )
    args = parser.parse_args()

    run_phase2_diagnostics(
        history_threshold=args.history_threshold,
        peer_threshold=args.peer_threshold
    )
