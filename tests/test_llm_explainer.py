"""Unit tests for Phase 4: Explanations, Validation, Usage Accounting, and Reproducibility."""

import json
from pathlib import Path
import pandas as pd
import pytest

from src.config import SHIPMENT_RECORDS_FILE, CONTEXT_NOTES_FILE
from src.data.loader import load_and_validate_shipments
from src.pipeline.weekly_aggregator import aggregate_weekly_shipments
from src.pipeline.baseline_calculator import compute_baselines_and_deviations
from src.pipeline.anomaly_detector import detect_candidate_anomalies
from src.rag.retriever import ContextNoteRetriever
from src.rag.guardrails import evaluate_all_candidates, EvidenceAssessment
from src.llm.cost_tracker import CostTracker, redact_secrets
from src.llm.prompts import (
    build_explanation_prompt,
    validate_structured_explanation,
)
from src.llm.explainer import (
    OfflineProvider,
    MockLLMProvider,
    ExplanationGenerator,
    generate_deterministic_explanation,
)


@pytest.fixture(scope="module")
def prepared_candidates():
    """Fixture providing candidates and validated evidence assessments."""
    df, _ = load_and_validate_shipments(SHIPMENT_RECORDS_FILE, strict=True)
    weekly = aggregate_weekly_shipments(df)
    enriched = compute_baselines_and_deviations(weekly)
    candidates = detect_candidate_anomalies(enriched, 20.0, 20.0)
    retriever = ContextNoteRetriever(CONTEXT_NOTES_FILE)
    evaluated_df, assessments = evaluate_all_candidates(candidates, retriever)
    return evaluated_df, assessments, retriever


def test_justified_explanation_cites_exact_note():
    """Verify that justified explanations cite the exact matched note ID."""
    text = generate_deterministic_explanation(
        route="Ahmedabad-Mumbai",
        week_of="2025-01-20",
        flagged="No (justified)",
        matched_note_id="N002"
    )
    assert "N002" in text
    assert "festival" in text.lower()
    assert "2025-01-20" in text


def test_unexplained_explanation_does_not_invent_cause():
    """Verify that unexplained candidates do not invent business causes or fake note IDs."""
    text = generate_deterministic_explanation(
        route="Delhi-Jaipur",
        week_of="2024-11-11",
        flagged="Yes",
        matched_note_id=""
    )
    assert "unexplained" in text.lower()
    assert "No matching note found" in text
    assert "N999" not in text


def test_validation_rejects_attempt_to_override_flagged():
    """Verify that if LLM tries to describe an unexplained candidate as justified, it is rejected."""
    bad_llm_json = json.dumps({"explanation": "This route increase is justified by note N002."})
    is_valid, _, reason = validate_structured_explanation(
        bad_llm_json,
        expected_flagged="Yes",
        expected_note_id=""
    )
    assert is_valid is False
    assert "erroneously described as justified" in reason


def test_validation_rejects_unsupported_or_invented_note_id():
    """Verify that if LLM hallucinates an unsupported note ID for an unexplained row, it is rejected."""
    bad_llm_json = json.dumps({"explanation": "A rate spike occurred due to note N999 on this corridor."})
    is_valid, _, reason = validate_structured_explanation(
        bad_llm_json,
        expected_flagged="Yes",
        expected_note_id=""
    )
    assert is_valid is False
    assert "unsupported note ID" in reason


def test_validation_rejects_malformed_json():
    """Verify that malformed JSON is rejected with informative error."""
    is_valid, _, reason = validate_structured_explanation(
        "Here is the explanation: costs rose because of weather.",
        expected_flagged="Yes",
        expected_note_id=""
    )
    assert is_valid is False
    assert "Malformed JSON" in reason


def test_provider_errors_trigger_fallback(prepared_candidates):
    """Verify that provider network/runtime errors trigger graceful deterministic fallback."""
    candidates_df, assessments, retriever = prepared_candidates
    mock_provider = MockLLMProvider(should_fail=True, error_message="Network Timeout 504")
    tracker = CostTracker()
    generator = ExplanationGenerator(mock_provider, tracker, retriever)

    df_out, tracker_out = generator.process_candidates(candidates_df, assessments)

    summary = tracker_out.get_summary()
    assert summary["error_calls"] == 20  # 20 candidate attempts failed
    # All 20 candidates must still receive deterministic fallback explanations
    cand_reasons = df_out[df_out["is_anomaly_candidate"]]["reason"]
    for r in cand_reasons:
        assert len(r) > 20
        assert "Network Timeout" not in r  # Error not leaked into final reason


def test_offline_mode_runs_with_zero_api_calls(prepared_candidates):
    """Verify offline mode executes completely with 0 API calls and $0.00 cost."""
    candidates_df, assessments, retriever = prepared_candidates
    offline_provider = OfflineProvider()
    tracker = CostTracker()
    generator = ExplanationGenerator(offline_provider, tracker, retriever)

    df_out, tracker_out = generator.process_candidates(candidates_df, assessments)

    summary = tracker_out.get_summary()
    assert summary["total_calls"] == 0
    assert summary["total_cost_usd"] == 0.0

    # Check that sample rows have exact expected offline explanations
    delhi_row = df_out[(df_out["route"] == "Delhi-Jaipur") & (df_out["week_of"] == "2024-11-11")].iloc[0]
    assert "No matching note found" in delhi_row["reason"]

    ahmedabad_row = df_out[(df_out["route"] == "Ahmedabad-Mumbai") & (df_out["week_of"] == "2025-01-20")].iloc[0]
    assert "Matches note N002" in ahmedabad_row["reason"]

    mumbai_row = df_out[(df_out["route"] == "Mumbai-Pune") & (df_out["week_of"] == "2025-09-15")].iloc[0]
    assert "closest note (N006" in mumbai_row["reason"]


def test_accurate_usage_and_cost_tracking_with_mock():
    """Verify accurate token counting and USD cost computation."""
    rates = {"mock": {"input_per_m": 0.15, "output_per_m": 0.60}}
    tracker = CostTracker(pricing_rates=rates)

    # 1000 input tokens, 200 output tokens:
    # cost = (1000 / 1M * 0.15) + (200 / 1M * 0.60) = 0.00015 + 0.00012 = 0.00027
    metric = tracker.record_call(
        provider="MockProvider",
        model="mock",
        input_tokens=1000,
        output_tokens=200,
        is_estimated=False
    )
    assert metric.total_tokens == 1200
    assert pytest.approx(metric.estimated_cost_usd, 0.000001) == 0.00027
    assert metric.is_estimated is False


def test_missing_usage_metadata_estimation():
    """Verify fallback token estimation when usage metadata is missing."""
    tracker = CostTracker()
    sample_text = "A" * 400  # 400 characters -> ~100 tokens
    est_tokens = tracker.estimate_tokens_from_text(sample_text)
    assert est_tokens == 100

    metric = tracker.record_call(
        provider="MockProvider",
        model="mock",
        input_tokens=est_tokens,
        output_tokens=25,
        is_estimated=True
    )
    assert metric.is_estimated is True
    summary = tracker.get_summary()
    assert summary["estimated_metadata_calls"] == 1


def test_secret_redaction_in_logs():
    """Verify that API keys and bearer tokens are redacted from logs and metric dumps."""
    secret_text = "Failed connecting with Authorization: Bearer sk-1234567890abcdef and AIzaSyD1234567890"
    redacted = redact_secrets(secret_text)
    assert "sk-1234567890abcdef" not in redacted
    assert "AIzaSyD1234567890" not in redacted
    assert "sk-***REDACTED***" in redacted
    assert "AIza***REDACTED***" in redacted


def test_deterministic_fallback_explanations_three_runs(prepared_candidates):
    """Verify three repeated runs in offline mode produce bit-for-bit identical explanations."""
    candidates_df, assessments, retriever = prepared_candidates

    gen1 = ExplanationGenerator(OfflineProvider(), CostTracker(), retriever)
    out1, _ = gen1.process_candidates(candidates_df, assessments)

    gen2 = ExplanationGenerator(OfflineProvider(), CostTracker(), retriever)
    out2, _ = gen2.process_candidates(candidates_df, assessments)

    gen3 = ExplanationGenerator(OfflineProvider(), CostTracker(), retriever)
    out3, _ = gen3.process_candidates(candidates_df, assessments)

    pd.testing.assert_series_equal(out1["reason"], out2["reason"])
    pd.testing.assert_series_equal(out1["reason"], out3["reason"])


def test_sample_verdicts_and_fields_remain_unchanged(prepared_candidates):
    """Verify that generating explanations does not modify numeric fields, flags, or note IDs."""
    candidates_df, assessments, retriever = prepared_candidates
    generator = ExplanationGenerator(OfflineProvider(), CostTracker(), retriever)
    out_df, _ = generator.process_candidates(candidates_df, assessments)

    # 1. Ahmedabad-Mumbai
    ahm = out_df[(out_df["route"] == "Ahmedabad-Mumbai") & (out_df["week_of"] == "2025-01-20")].iloc[0]
    assert ahm["cost_per_tonne_km_rounded"] == 3.29
    assert ahm["flagged"] == "No (justified)"
    assert ahm["matched_note_id"] == "N002"

    # 2. Delhi-Jaipur
    delhi = out_df[(out_df["route"] == "Delhi-Jaipur") & (out_df["week_of"] == "2024-11-11")].iloc[0]
    assert delhi["cost_per_tonne_km_rounded"] == 4.17
    assert delhi["flagged"] == "Yes"
    assert delhi["matched_note_id"] == ""

    # 3. Mumbai-Pune
    mumbai = out_df[(out_df["route"] == "Mumbai-Pune") & (out_df["week_of"] == "2025-09-15")].iloc[0]
    assert mumbai["cost_per_tonne_km_rounded"] == 3.98
    assert mumbai["flagged"] == "Yes"
    assert mumbai["matched_note_id"] == ""
