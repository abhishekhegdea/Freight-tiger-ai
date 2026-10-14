"""Diagnostic CLI script for Phase 4: Explanations and Token/Cost Accounting."""

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
    CONTEXT_NOTES_FILE,
    RUN_METRICS_FILE,
    DEFAULT_HISTORY_THRESHOLD,
    DEFAULT_PEER_THRESHOLD
)
from src.data.loader import load_and_validate_shipments
from src.pipeline.weekly_aggregator import aggregate_weekly_shipments
from src.pipeline.baseline_calculator import compute_baselines_and_deviations
from src.pipeline.anomaly_detector import detect_candidate_anomalies
from src.rag.retriever import ContextNoteRetriever
from src.rag.guardrails import evaluate_all_candidates
from src.llm.cost_tracker import CostTracker
from src.llm.explainer import (
    ExplanationGenerator,
    get_llm_provider,
    OfflineProvider,
)


def run_phase4_diagnostics(provider_name: str = "offline", model_name: str = "default"):
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 85)
    print("FREIGHTTIGER PHASE 4: EXPLANATIONS & AUDITABLE TOKEN/COST ACCOUNTING")
    print(f"Mode: Provider='{provider_name}' | Model='{model_name}'")
    print("=" * 85)

    # 1. Pipeline execution up to Phase 3
    df, _ = load_and_validate_shipments(SHIPMENT_RECORDS_FILE, strict=True)
    weekly = aggregate_weekly_shipments(df)
    enriched = compute_baselines_and_deviations(weekly)
    candidates = detect_candidate_anomalies(
        enriched,
        history_threshold=DEFAULT_HISTORY_THRESHOLD,
        peer_threshold=DEFAULT_PEER_THRESHOLD
    )

    retriever = ContextNoteRetriever(CONTEXT_NOTES_FILE)
    evaluated_df, assessments = evaluate_all_candidates(candidates, retriever)

    # 2. Explanations and Usage Tracking
    provider = get_llm_provider(provider_name, model_name)
    cost_tracker = CostTracker()
    generator = ExplanationGenerator(provider, cost_tracker, retriever)

    explained_df, tracker = generator.process_candidates(evaluated_df, assessments)

    # Persist run_metrics.json
    metrics_path = tracker.save_run_metrics(RUN_METRICS_FILE)
    summary = tracker.get_summary()

    candidates_only = explained_df[explained_df["is_anomaly_candidate"]].copy()

    print("\n--- Model Usage & Accounting Summary ---")
    print(f"Candidate explanations generated : {len(candidates_only)}")
    print(f"Total LLM API calls made         : {summary['total_calls']}")
    print(f"  - Successful API calls         : {summary['successful_calls']}")
    print(f"  - Fallback / Rejected calls    : {summary['fallback_calls']}")
    print(f"  - Error calls                  : {summary['error_calls']}")
    print(f"Token counts:")
    print(f"  - Input tokens                 : {summary['total_input_tokens']}")
    print(f"  - Output tokens                : {summary['total_output_tokens']}")
    print(f"  - Total tokens                 : {summary['total_tokens']}")
    print(f"  - Provider metadata calls      : {summary['actual_metadata_calls']} (actual)")
    print(f"  - Estimated metadata calls     : {summary['estimated_metadata_calls']} (estimated)")
    print(f"Estimated Cost in USD            : ${summary['total_cost_usd']:.6f}")
    print(f"Audit log persisted to           : {metrics_path}")

    # 3. Official Sample Cases Explanations
    print("\n" + "=" * 85)
    print("OFFICIAL SAMPLE ROWS EXPLANATION AUDIT")
    print("=" * 85)
    sample_cases = [
        ("Delhi-Jaipur", "2024-11-11"),
        ("Ahmedabad-Mumbai", "2025-01-20"),
        ("Mumbai-Pune", "2025-09-15"),
    ]
    for route, week_of in sample_cases:
        row = explained_df[(explained_df["route"] == route) & (explained_df["week_of"] == week_of)].iloc[0]
        print(f"\n[{route} | {week_of}]")
        print(f"  Cost/t-km : {row['cost_per_tonne_km_rounded']:.2f}")
        print(f"  History   : {row['vs_own_history']}")
        print(f"  Peers     : {row['vs_similar_routes']}")
        print(f"  Flagged   : {row['flagged']}")
        print(f"  Note ID   : '{row['matched_note_id']}'")
        print(f"  Reason    : \"{row['reason']}\"")

    # 4. Shortlist of all candidate explanations
    print("\n" + "=" * 85)
    print("ALL 20 CANDIDATE EXPLANATIONS PREVIEW")
    print("=" * 85)
    for _, r in candidates_only.iterrows():
        note_str = f"[{r['matched_note_id']}]" if r['matched_note_id'] else "[No Note]"
        print(f"- {r['route']:<18} ({r['week_of']}) | {r['flagged']:<15} {note_str:<10} | \"{r['reason'][:75]}...\"")

    print("\n" + "=" * 85)
    print("PHASE 4 DIAGNOSTICS COMPLETE")
    print("=" * 85)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FreightTiger Phase 4: Explanations & Token Accounting Diagnostics")
    parser.add_argument(
        "--provider",
        type=str,
        default="offline",
        help="LLM provider: 'offline' (default, no API key), 'mock', 'openai', 'groq'"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="default",
        help="Model identifier (e.g. gpt-4o-mini, llama-3.1-8b)"
    )
    args = parser.parse_args()

    run_phase4_diagnostics(provider_name=args.provider, model_name=args.model)
