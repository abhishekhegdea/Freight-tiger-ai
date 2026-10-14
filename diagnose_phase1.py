"""Diagnostic CLI script for Phase 1 verification against official sample output."""

import sys
from pathlib import Path
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SHIPMENT_RECORDS_FILE
from src.data.loader import load_and_validate_shipments
from src.pipeline.weekly_aggregator import aggregate_weekly_shipments
from src.pipeline.baseline_calculator import compute_baselines_and_deviations


EXPECTED_SAMPLES = [
    {
        "route": "Ahmedabad-Mumbai",
        "week_of": "2025-01-20",
        "expected_cost": 3.29,
        "expected_history_pct": 29.5,
        "expected_peer_pct": 22.5,
        "expected_vs_history": "+29.5% vs this route's past average",
        "expected_vs_peers": "+22.5% vs similar-length routes this week",
    },
    {
        "route": "Delhi-Jaipur",
        "week_of": "2024-11-11",
        "expected_cost": 4.17,
        "expected_history_pct": 35.5,
        "expected_peer_pct": 21.0,
        "expected_vs_history": "+35.5% vs this route's past average",
        "expected_vs_peers": "+21.0% vs similar-length routes this week",
    },
    {
        "route": "Mumbai-Pune",
        "week_of": "2025-09-15",
        "expected_cost": 3.98,
        "expected_history_pct": 9.2,
        "expected_peer_pct": 23.6,
        "expected_vs_history": "+9.2% vs this route's past average",
        "expected_vs_peers": "+23.6% vs similar-length routes this week",
    },
]


def run_diagnostics(file_path: Path = SHIPMENT_RECORDS_FILE) -> bool:
    print("=" * 80)
    print("FREIGHTTIGER PHASE 1: DETERMINISTIC PIPELINE DIAGNOSTICS")
    print("=" * 80)

    # 1. Load and validate
    print(f"\n[1/3] Loading and validating: {file_path} ...")
    df, report = load_and_validate_shipments(file_path, strict=True)
    print(f"      Rows read: {report['total_rows_read']} | Status: {report['status']}")

    # 2. Aggregate weekly
    print("\n[2/3] Aggregating weekly shipments (Monday to Sunday) ...")
    weekly_df = aggregate_weekly_shipments(df)
    print(f"      Total route-week records: {len(weekly_df)}")

    # 3. Compute baselines
    print("\n[3/3] Computing trailing 8-week history and same-route-type peer baselines ...")
    enriched_df = compute_baselines_and_deviations(weekly_df)

    print("\n" + "=" * 80)
    print("VERIFICATION AGAINST OFFICIAL SAMPLE ROWS")
    print("=" * 80)

    all_passed = True

    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    for sample in EXPECTED_SAMPLES:
        route = sample["route"]
        week_of = sample["week_of"]
        match = enriched_df[(enriched_df["route"] == route) & (enriched_df["week_of"] == week_of)]

        if match.empty:
            print(f"\n[FAIL] Missing record for {route} on week {week_of}")
            all_passed = False
            continue

        row = match.iloc[0]
        actual_cost = row["cost_per_tonne_km_rounded"]
        actual_cost_unrounded = row["cost_per_tonne_km"]
        actual_history_pct = row["pct_vs_history"]
        actual_peer_pct = row["pct_vs_peer"]
        actual_vs_history = row["vs_own_history"]
        actual_vs_peers = row["vs_similar_routes"]

        cost_diff = abs(actual_cost - sample["expected_cost"])
        hist_diff = abs(actual_history_pct - sample["expected_history_pct"])
        peer_diff = abs(actual_peer_pct - sample["expected_peer_pct"])

        cost_pass = cost_diff <= 0.01
        hist_pass = hist_diff <= 0.1 and actual_vs_history == sample["expected_vs_history"]
        peer_pass = peer_diff <= 0.1 and actual_vs_peers == sample["expected_vs_peers"]

        status = "[PASS]" if (cost_pass and hist_pass and peer_pass) else "[FAIL]"
        if not (cost_pass and hist_pass and peer_pass):
            all_passed = False

        print(f"\n--- {route} ({week_of}) [{status}] ---")
        print(f"  cost_per_tonne_km : Actual = {actual_cost:.2f} (raw {actual_cost_unrounded:.4f}) | Expected = {sample['expected_cost']:.2f} | Diff = {cost_diff:.4f}")
        print(f"  vs_own_history    : Actual = '{actual_vs_history}' (raw {actual_history_pct:+.2f}%) | Expected = '{sample['expected_vs_history']}' | Diff = {hist_diff:.2f}%")
        print(f"  vs_similar_routes : Actual = '{actual_vs_peers}' (raw {actual_peer_pct:+.2f}%) | Expected = '{sample['expected_vs_peers']}' | Diff = {peer_diff:.2f}%")
        print(f"  Prior weeks used  : {row['history_prior_weeks_count']} weeks")
        print(f"  Peer routes count : {row['peer_routes_count']} peers")

    print("\n" + "=" * 80)
    if all_passed:
        print("RESULT: ALL 3 SAMPLE ROWS MATCH THE CONTRACT EXACTLY (WITHIN TOLERANCE)")
    else:
        print("RESULT: ONE OR MORE SAMPLES FAILED VERIFICATION")
    print("=" * 80)

    return all_passed


if __name__ == "__main__":
    success = run_diagnostics()
    sys.exit(0 if success else 1)
