"""FreightTiger Freight-Cost Anomaly Detection & AI Explanation Pipeline Entry Point.

Connects:
1. Data loading and validation.
2. Weekly aggregation and metric calculations.
3. Historical (trailing 8-week) and peer baseline calculations.
4. Deterministic anomaly detection.
5. Context retrieval and evidence validation guardrails.
6. Explanation generation (deterministic offline or optional LLM).
7. Final RFC 4180 compliant CSV export and token/cost accounting.
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    SHIPMENT_RECORDS_FILE,
    CONTEXT_NOTES_FILE,
    FINAL_OUTPUT_FILE,
    RUN_METRICS_FILE,
    DEFAULT_HISTORY_THRESHOLD,
    DEFAULT_PEER_THRESHOLD,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_LLM_MODEL,
)
from src.data.loader import load_and_validate_shipments
from src.pipeline.weekly_aggregator import aggregate_weekly_shipments
from src.pipeline.baseline_calculator import compute_baselines_and_deviations
from src.pipeline.anomaly_detector import detect_candidate_anomalies, get_anomaly_summary
from src.pipeline.exporter import format_and_export_csv, SUBMISSION_COLUMNS
from src.rag.retriever import ContextNoteRetriever
from src.rag.guardrails import evaluate_all_candidates
from src.llm.cost_tracker import CostTracker
from src.llm.explainer import ExplanationGenerator, get_llm_provider


def execute_pipeline(
    shipments_path: Path | str = SHIPMENT_RECORDS_FILE,
    notes_path: Path | str = CONTEXT_NOTES_FILE,
    output_csv_path: Path | str = FINAL_OUTPUT_FILE,
    metrics_path: Path | str = RUN_METRICS_FILE,
    provider_name: str = DEFAULT_LLM_PROVIDER,
    model_name: str = DEFAULT_LLM_MODEL,
    history_threshold: float = DEFAULT_HISTORY_THRESHOLD,
    peer_threshold: float = DEFAULT_PEER_THRESHOLD,
    include_all_weeks: bool = False
) -> Tuple[pd.DataFrame, Path, Path]:
    """Runs the end-to-end anomaly detection and explanation pipeline."""
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 85)
    print("FREIGHTTIGER FREIGHT-COST ANOMALY DETECTION PIPELINE")
    print("=" * 85)
    print(f"Provider: '{provider_name}' | Model: '{model_name}' | Mode: {'All Weeks' if include_all_weeks else 'Anomalies Only (Default)'}")
    print(f"Thresholds: History >= {history_threshold:.1f}% | Peer >= {peer_threshold:.1f}%")

    # Step 1: Load and Validate
    print("\n[Step 1/6] Ingesting and validating shipment records...")
    raw_df, report = load_and_validate_shipments(shipments_path, strict=True)
    print(f"           Validated {report['total_rows_read']} raw shipments. (Status: {report['status']})")

    # Step 2: Weekly Aggregation
    print("\n[Step 2/6] Aggregating shipments into Monday-to-Sunday weeks...")
    weekly_df = aggregate_weekly_shipments(raw_df)
    print(f"           Aggregated into {len(weekly_df)} route-week records across {weekly_df['route'].nunique()} routes.")

    # Step 3: Compute Baselines
    print("\n[Step 3/6] Computing trailing 8-week history and same-route-type peer baselines...")
    enriched_df = compute_baselines_and_deviations(weekly_df)

    # Step 4: Detect Candidate Anomalies
    print("\n[Step 4/6] Running deterministic anomaly detector...")
    candidates_df = detect_candidate_anomalies(
        enriched_df,
        history_threshold=history_threshold,
        peer_threshold=peer_threshold
    )
    summary = get_anomaly_summary(candidates_df)
    print(f"           Identified {summary['total_candidates']} candidate anomalies ({summary['candidate_rate_pct']}% of records).")

    # Step 5: RAG Context Retrieval & Evidence Validation
    print("\n[Step 5/6] Retrieving context notes & enforcing anti-hallucination guardrails...")
    retriever = ContextNoteRetriever(notes_path)
    evaluated_df, assessments = evaluate_all_candidates(candidates_df, retriever)

    justified_count = sum(1 for a in assessments if a.selected_note_id)
    unexplained_count = sum(1 for a in assessments if not a.selected_note_id)
    print(f"           Assessed {len(assessments)} candidates: {justified_count} Justified, {unexplained_count} Unexplained.")

    # Step 6: Generate Explanations & Accounting
    print(f"\n[Step 6/6] Generating explanations using provider '{provider_name}'...")
    provider = get_llm_provider(provider_name, model_name)
    cost_tracker = CostTracker()
    generator = ExplanationGenerator(provider, cost_tracker, retriever)

    final_df, tracker = generator.process_candidates(evaluated_df, assessments)

    # Persist run metrics
    saved_metrics = tracker.save_run_metrics(metrics_path)
    metrics_summary = tracker.get_summary()

    # Export Submission CSV
    exported_csv = format_and_export_csv(
        final_df,
        output_path=output_csv_path,
        include_all_weeks=include_all_weeks
    )

    print("\n" + "=" * 85)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 85)
    print(f"Output CSV written to      : {exported_csv}")
    print(f"  - Total rows exported    : {len(pd.read_csv(exported_csv))}")
    print(f"  - Total columns          : {len(SUBMISSION_COLUMNS)}")
    print(f"Run metrics written to     : {saved_metrics}")
    print(f"  - Total LLM API calls    : {metrics_summary['total_calls']}")
    print(f"  - Input / Output tokens  : {metrics_summary['total_input_tokens']} in / {metrics_summary['total_output_tokens']} out")
    print(f"  - Estimated USD spend    : ${metrics_summary['total_cost_usd']:.6f}")

    return final_df, exported_csv, saved_metrics


if __name__ == "__main__":
    from typing import Tuple

    parser = argparse.ArgumentParser(description="FreightTiger Anomaly Detection & AI Explanation Pipeline")
    parser.add_argument("--shipments", type=str, default=str(SHIPMENT_RECORDS_FILE), help="Path to shipment_records.csv")
    parser.add_argument("--notes", type=str, default=str(CONTEXT_NOTES_FILE), help="Path to context_notes.csv")
    parser.add_argument("--output", type=str, default=str(FINAL_OUTPUT_FILE), help="Path to output CSV")
    parser.add_argument("--metrics", type=str, default=str(RUN_METRICS_FILE), help="Path to run_metrics.json")
    parser.add_argument("--provider", type=str, default=DEFAULT_LLM_PROVIDER, help="Provider: offline, mock, openai, groq")
    parser.add_argument("--model", type=str, default=DEFAULT_LLM_MODEL, help="Model name")
    parser.add_argument("--history-threshold", type=float, default=DEFAULT_HISTORY_THRESHOLD, help="History threshold (%)")
    parser.add_argument("--peer-threshold", type=float, default=DEFAULT_PEER_THRESHOLD, help="Peer threshold (%)")
    parser.add_argument("--include-all-weeks", action="store_true", help="Include all 728 route-weeks in CSV output")

    args = parser.parse_args()

    execute_pipeline(
        shipments_path=args.shipments,
        notes_path=args.notes,
        output_csv_path=args.output,
        metrics_path=args.metrics,
        provider_name=args.provider,
        model_name=args.model,
        history_threshold=args.history_threshold,
        peer_threshold=args.peer_threshold,
        include_all_weeks=args.include_all_weeks
    )
