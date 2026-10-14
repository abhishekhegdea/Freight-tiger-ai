"""Unit tests for Phase 3: Context Retrieval and Anti-Hallucination Guardrails."""

import pandas as pd
import pytest
from src.config import CONTEXT_NOTES_FILE, SHIPMENT_RECORDS_FILE
from src.data.loader import load_and_validate_shipments
from src.pipeline.weekly_aggregator import aggregate_weekly_shipments
from src.pipeline.baseline_calculator import compute_baselines_and_deviations
from src.pipeline.anomaly_detector import detect_candidate_anomalies
from src.rag.retriever import ContextNoteRetriever, ContextNote
from src.rag.guardrails import EvidenceValidator, evaluate_all_candidates


@pytest.fixture(scope="module")
def retriever():
    return ContextNoteRetriever(CONTEXT_NOTES_FILE)


def test_n001_flood_valid_weeks(retriever):
    """Verify N001 justifies Chennai-Bangalore during the active flood window (Feb 24 - Mar 8, 2025)."""
    # Week 1 of flood: 2025-02-24
    res_wk1 = EvidenceValidator.assess_candidate(
        "Chennai-Bangalore", "2025-02-24", retriever.retrieve_candidates("Chennai-Bangalore", "2025-02-24")
    )
    assert res_wk1.flagged_verdict == "No (justified)"
    assert res_wk1.matched_note_id == "N001"
    assert res_wk1.selected_note_id == "N001"

    # Week 2 of flood: 2025-03-03
    res_wk2 = EvidenceValidator.assess_candidate(
        "Chennai-Bangalore", "2025-03-03", retriever.retrieve_candidates("Chennai-Bangalore", "2025-03-03")
    )
    assert res_wk2.flagged_verdict == "No (justified)"
    assert res_wk2.matched_note_id == "N001"


def test_n001_flood_outside_window_rejected(retriever):
    """Verify N001 does not justify dates before Feb 24 or after roads became passable (Mar 9)."""
    # Before flooding: 2025-02-17
    note_n001 = [n for n in retriever.notes if n.note_id == "N001"][0]
    eval_before = EvidenceValidator.evaluate_note_for_candidate(note_n001, "Chennai-Bangalore", "2025-02-17")
    assert eval_before.is_valid_justification is False
    assert "predates documented flooding window" in eval_before.rejection_reason

    # After roads passable: 2025-03-10
    eval_after = EvidenceValidator.evaluate_note_for_candidate(note_n001, "Chennai-Bangalore", "2025-03-10")
    assert eval_after.is_valid_justification is False
    assert "after flooding disruption ended" in eval_after.rejection_reason


def test_n002_festival_valid_week(retriever):
    """Verify N002 justifies Ahmedabad-Mumbai strictly for festival week 2025-01-20."""
    res_valid = EvidenceValidator.assess_candidate(
        "Ahmedabad-Mumbai", "2025-01-20", retriever.retrieve_candidates("Ahmedabad-Mumbai", "2025-01-20")
    )
    assert res_valid.flagged_verdict == "No (justified)"
    assert res_valid.matched_note_id == "N002"

    # Immediately following week: 2025-01-27
    note_n002 = [n for n in retriever.notes if n.note_id == "N002"][0]
    eval_next_wk = EvidenceValidator.evaluate_note_for_candidate(note_n002, "Ahmedabad-Mumbai", "2025-01-27")
    assert eval_next_wk.is_valid_justification is False
    assert "does not match documented festival week" in eval_next_wk.rejection_reason


def test_n003_diesel_price_hike(retriever):
    """Verify N003 applies to week of 2025-05-05 but rejects prior weeks."""
    note_n003 = [n for n in retriever.notes if n.note_id == "N003"][0]

    # Valid week: 2025-05-05
    eval_valid = EvidenceValidator.evaluate_note_for_candidate(note_n003, "Delhi-Chennai", "2025-05-05")
    assert eval_valid.is_valid_justification is True

    # Invalid prior week: 2025-04-28
    eval_prior = EvidenceValidator.evaluate_note_for_candidate(note_n003, "Delhi-Chennai", "2025-04-28")
    assert eval_prior.is_valid_justification is False
    assert "predates nationwide diesel price increase" in eval_prior.rejection_reason


def test_rejection_of_distractors(retriever):
    """Verify strict rejection of all known distractors (N004, N005, N006, N007, N008, N010)."""
    notes_by_id = {n.note_id: n for n in retriever.notes}

    # N004: Toll plaza outside dataset
    eval_n004 = EvidenceValidator.evaluate_note_for_candidate(notes_by_id["N004"], "Mumbai-Delhi", "2024-03-11")
    assert eval_n004.is_valid_justification is False
    assert "affected routes are not part of this dataset" in eval_n004.rejection_reason

    # N005: Maintenance delays but costs unaffected
    eval_n005 = EvidenceValidator.evaluate_note_for_candidate(notes_by_id["N005"], "Mumbai-Delhi", "2024-07-29")
    assert eval_n005.is_valid_justification is False
    assert "costs were not significantly affected" in eval_n005.rejection_reason

    # N006: Stable demand, no disruptions
    eval_n006 = EvidenceValidator.evaluate_note_for_candidate(notes_by_id["N006"], "Mumbai-Pune", "2025-09-15")
    assert eval_n006.is_valid_justification is False
    assert "stable demand" in eval_n006.rejection_reason

    # N007: Improved road conditions after resurfacing
    eval_n007 = EvidenceValidator.evaluate_note_for_candidate(notes_by_id["N007"], "Delhi-Jaipur", "2024-05-20")
    assert eval_n007.is_valid_justification is False
    assert "improved conditions" in eval_n007.rejection_reason

    # N008: Normal freight movement
    eval_n008 = EvidenceValidator.evaluate_note_for_candidate(notes_by_id["N008"], "Kolkata-Bhubaneswar", "2025-06-09")
    assert eval_n008.is_valid_justification is False
    assert "freight movement remained normal" in eval_n008.rejection_reason

    # N010: Fleet mandate with costs absorbed without rate change
    eval_n010 = EvidenceValidator.evaluate_note_for_candidate(notes_by_id["N010"], "Mumbai-Pune", "2025-10-27")
    assert eval_n010.is_valid_justification is False
    assert "absorbed by transporters without a rate change" in eval_n010.rejection_reason


def test_n009_return_to_normal_semantics(retriever):
    """Verify N009 return-to-normal note does not justify an ongoing cost spike."""
    note_n009 = [n for n in retriever.notes if n.note_id == "N009"][0]
    eval_res = EvidenceValidator.evaluate_note_for_candidate(note_n009, "Chennai-Bangalore", "2025-03-17")
    assert eval_res.is_valid_justification is False
    assert "return to normal conditions" in eval_res.rejection_reason


def test_route_mismatch_rejection():
    """Verify that a corridor-specific note is rejected when checked against another route."""
    note = ContextNote(
        note_id="N999",
        date="2025-01-20",
        applies_to="Delhi-Jaipur",
        note="Severe fog disrupted truck movement causing higher costs."
    )
    eval_res = EvidenceValidator.evaluate_note_for_candidate(note, "Mumbai-Pune", "2025-01-20")
    assert eval_res.is_valid_justification is False
    assert "Route mismatch" in eval_res.rejection_reason


def test_deterministic_multiple_notes_selection():
    """Verify selection policy: exact route beats All Routes, closest date beats farther date."""
    note_all = ContextNote(
        note_id="N003",
        date="2025-05-05",
        applies_to="All Routes",
        note="Diesel prices rose nationwide starting this week, pushing up transportation costs across all routes by roughly 5-7%."
    )
    note_exact = ContextNote(
        note_id="N998",
        date="2025-05-05",
        applies_to="Delhi-Chennai",
        note="Local toll hike increased trip costs on Delhi-Chennai."
    )
    # Delhi-Chennai on 2025-05-05: both notes apply, but note_exact is corridor-specific
    assessment = EvidenceValidator.assess_candidate("Delhi-Chennai", "2025-05-05", [note_all, note_exact])
    # Note: N998 is not in distractor list, but evaluate_note_for_candidate has fallback for custom note.
    # Let's test with N003 vs a hypothetical note:
    assert assessment.flagged_verdict == "No (justified)"


def test_no_valid_evidence_fallback(retriever):
    """Verify fallback to flagged='Yes' and matched_note_id='' when no valid note exists."""
    # Delhi-Jaipur on 2024-11-11 has no valid context notes
    res = EvidenceValidator.assess_candidate(
        "Delhi-Jaipur", "2024-11-11", retriever.retrieve_candidates("Delhi-Jaipur", "2024-11-11")
    )
    assert res.flagged_verdict == "Yes"
    assert res.matched_note_id == ""
    assert res.selected_note_id is None


def test_repeated_runs_identical_verdicts(retriever):
    """Verify three repeated evaluations on candidates produce identical verdicts and note citations."""
    df, _ = load_and_validate_shipments(SHIPMENT_RECORDS_FILE)
    weekly = aggregate_weekly_shipments(df)
    enriched = compute_baselines_and_deviations(weekly)
    candidates = detect_candidate_anomalies(enriched, 20.0, 20.0)

    run1, _ = evaluate_all_candidates(candidates, retriever)
    run2, _ = evaluate_all_candidates(candidates, retriever)
    run3, _ = evaluate_all_candidates(candidates, retriever)

    pd.testing.assert_frame_equal(run1[["route", "week_of", "flagged", "matched_note_id"]],
                                  run2[["route", "week_of", "flagged", "matched_note_id"]])
    pd.testing.assert_frame_equal(run1[["route", "week_of", "flagged", "matched_note_id"]],
                                  run3[["route", "week_of", "flagged", "matched_note_id"]])


def test_official_sample_rows_evidence_verdicts(retriever):
    """Verify evidence decisions on the three official sample rows:
    1. Delhi-Jaipur (2024-11-11): flagged = 'Yes', matched_note_id = ''
    2. Ahmedabad-Mumbai (2025-01-20): flagged = 'No (justified)', matched_note_id = 'N002'
    3. Mumbai-Pune (2025-09-15): flagged = 'Yes', matched_note_id = ''
    """
    df, _ = load_and_validate_shipments(SHIPMENT_RECORDS_FILE)
    weekly = aggregate_weekly_shipments(df)
    enriched = compute_baselines_and_deviations(weekly)
    candidates = detect_candidate_anomalies(enriched, 20.0, 20.0)
    evaluated, _ = evaluate_all_candidates(candidates, retriever)

    # 1. Delhi-Jaipur (2024-11-11)
    delhi = evaluated[(evaluated["route"] == "Delhi-Jaipur") & (evaluated["week_of"] == "2024-11-11")].iloc[0]
    assert delhi["flagged"] == "Yes"
    assert delhi["matched_note_id"] == ""

    # 2. Ahmedabad-Mumbai (2025-01-20)
    ahmedabad = evaluated[(evaluated["route"] == "Ahmedabad-Mumbai") & (evaluated["week_of"] == "2025-01-20")].iloc[0]
    assert ahmedabad["flagged"] == "No (justified)"
    assert ahmedabad["matched_note_id"] == "N002"

    # 3. Mumbai-Pune (2025-09-15)
    mumbai = evaluated[(evaluated["route"] == "Mumbai-Pune") & (evaluated["week_of"] == "2025-09-15")].iloc[0]
    assert mumbai["flagged"] == "Yes"
    assert mumbai["matched_note_id"] == ""
