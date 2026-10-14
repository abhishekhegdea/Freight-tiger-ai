"""Diagnostic CLI script for Phase 3: Context Retrieval & Anti-Hallucination Guardrails."""

import sys
from pathlib import Path
from collections import Counter
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    SHIPMENT_RECORDS_FILE,
    CONTEXT_NOTES_FILE,
    DEFAULT_HISTORY_THRESHOLD,
    DEFAULT_PEER_THRESHOLD
)
from src.data.loader import load_and_validate_shipments
from src.pipeline.weekly_aggregator import aggregate_weekly_shipments
from src.pipeline.baseline_calculator import compute_baselines_and_deviations
from src.pipeline.anomaly_detector import detect_candidate_anomalies
from src.rag.retriever import ContextNoteRetriever
from src.rag.guardrails import EvidenceValidator, evaluate_all_candidates


def run_phase3_diagnostics():
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 85)
    print("FREIGHTTIGER PHASE 3: CONTEXT RETRIEVAL & ANTI-HALLUCINATION GUARDRAILS")
    print("=" * 85)

    # 1. Pipeline execution up to Phase 2
    df, _ = load_and_validate_shipments(SHIPMENT_RECORDS_FILE, strict=True)
    weekly = aggregate_weekly_shipments(df)
    enriched = compute_baselines_and_deviations(weekly)
    candidates = detect_candidate_anomalies(
        enriched,
        history_threshold=DEFAULT_HISTORY_THRESHOLD,
        peer_threshold=DEFAULT_PEER_THRESHOLD
    )

    # 2. Phase 3 Context retrieval & evidence evaluation
    retriever = ContextNoteRetriever(CONTEXT_NOTES_FILE)
    evaluated_df, assessments = evaluate_all_candidates(candidates, retriever)

    # Filter to candidate rows only
    candidate_assessments = [a for a in assessments]
    total_candidates = len(candidate_assessments)
    justified = [a for a in candidate_assessments if a.selected_note_id is not None]
    unexplained = [a for a in candidate_assessments if a.selected_note_id is None]

    print("\n--- Assessment Volumes ---")
    print(f"Total candidate anomalies assessed : {total_candidates}")
    print(f"  - Justified (valid note match)   : {len(justified)} (flagged = 'No (justified)')")
    print(f"  - Unexplained (no valid evidence): {len(unexplained)} (flagged = 'Yes')")

    # Counts of selected note IDs
    note_counts = Counter([a.selected_note_id for a in justified])
    print("\n--- Counts of Selected Note IDs ---")
    for note_id, count in note_counts.most_common():
        print(f"  - Note {note_id}: {count} candidate(s) justified")

    # Retrieved-but-rejected notes and reasons
    rejections: List[tuple] = []
    for a in candidate_assessments:
        for ev in a.evaluations:
            if not ev.is_valid_justification:
                rejections.append((ev.note_id, a.candidate_route, a.candidate_week_of, ev.rejection_reason))

    print(f"\n--- Retrieved-but-Rejected Notes ({len(rejections)} instances across candidates) ---")
    rejection_reasons_counter = Counter([r[3] for r in rejections])
    for reason, count in rejection_reasons_counter.most_common():
        print(f"  [{count}x] {reason}")

    # Official sample cases evidence check
    print("\n--- Official Sample Cases Evidence Verdicts ---")
    sample_cases = [
        ("Delhi-Jaipur", "2024-11-11", "Yes", ""),
        ("Ahmedabad-Mumbai", "2025-01-20", "No (justified)", "N002"),
        ("Mumbai-Pune", "2025-09-15", "Yes", ""),
    ]
    all_sample_pass = True
    for route, week_of, exp_flagged, exp_note in sample_cases:
        row = evaluated_df[(evaluated_df["route"] == route) & (evaluated_df["week_of"] == week_of)].iloc[0]
        act_flagged = row["flagged"]
        act_note = row["matched_note_id"]
        status = "[PASS]" if (act_flagged == exp_flagged and act_note == exp_note) else "[FAIL]"
        if status == "[FAIL]":
            all_sample_pass = False

        # Find assessment details
        ass = next((a for a in candidate_assessments if a.candidate_route == route and a.candidate_week_of == week_of), None)
        retrieved_ids = ass.retrieved_note_ids if ass else []
        print(f"\n  {route} ({week_of}) [{status}]")
        print(f"    flagged:         Actual='{act_flagged}' | Expected='{exp_flagged}'")
        print(f"    matched_note_id: Actual='{act_note}' | Expected='{exp_note}'")
        print(f"    Retrieved notes: {retrieved_ids}")
        if ass and ass.evaluations:
            for ev in ass.evaluations:
                val_status = "VALID" if ev.is_valid_justification else "REJECTED"
                print(f"      - Note {ev.note_id} ({ev.applies_to}, {ev.date}): [{val_status}] {ev.rejection_reason or 'Supported'}")

    # Full Table of all candidates and their evidence status
    print("\n" + "=" * 85)
    print("DETAILED EVIDENCE TABLE FOR ALL 20 CANDIDATES")
    print("=" * 85)
    header = f"{'Route':<20} {'Week Of':<12} {'Cost':<6} {'Hist %':<9} {'Peer %':<9} {'Flagged':<16} {'Note':<6} {'Status':<12}"
    print(header)
    print("-" * len(header))
    for a in candidate_assessments:
        r = evaluated_df[(evaluated_df["route"] == a.candidate_route) & (evaluated_df["week_of"] == a.candidate_week_of)].iloc[0]
        hist_s = f"{r['pct_vs_history']:+.1f}%" if pd.notna(r['pct_vs_history']) else "N/A"
        peer_s = f"{r['pct_vs_peer']:+.1f}%" if pd.notna(r['pct_vs_peer']) else "N/A"
        print(f"{a.candidate_route:<20} {a.candidate_week_of:<12} {r['cost_per_tonne_km_rounded']:<6.2f} "
              f"{hist_s:<9} {peer_s:<9} {r['flagged']:<16} {r['matched_note_id']:<6} {r['evidence_status']:<12}")

    print("\n" + "=" * 85)
    print("PHASE 3 DIAGNOSTICS COMPLETE")
    print("=" * 85)
    return all_sample_pass


if __name__ == "__main__":
    success = run_phase3_diagnostics()
    sys.exit(0 if success else 1)
