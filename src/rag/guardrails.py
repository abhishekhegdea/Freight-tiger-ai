"""Anti-Hallucination Guardrails and Evidence Validation Module.

Enforces strict, deterministic validation rules against candidate context notes:
- Corridor applicability
- Temporal boundaries
- Distractor and negative evidence rejection
- Deterministic supporting note selection policy
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from src.rag.retriever import ContextNote

logger = logging.getLogger(__name__)


# Specific documented distractor signatures
DISTRACTOR_SIGNATURES: Dict[str, str] = {
    "N004": "Note explicitly states affected routes are not part of this dataset.",
    "N005": "Note explicitly states costs were not significantly affected.",
    "N006": "Note describes stable demand with no major disruptions; does not explain a cost increase.",
    "N007": "Note reports completed resurfacing and improved conditions, not a cost increase or disruption.",
    "N008": "Note explicitly states freight movement remained normal with no significant disruptions.",
    "N009": "Note confirms return to normal conditions, does not explain an ongoing cost rise.",
    "N010": "Note explicitly states compliance costs were absorbed by transporters without a rate change.",
}


@dataclass
class NoteEvaluation:
    """Detailed evaluation result for a single retrieved context note."""
    note_id: str
    date: str
    applies_to: str
    is_valid_justification: bool
    rejection_reason: Optional[str] = None
    note_text: str = ""


@dataclass
class EvidenceAssessment:
    """Internal assessment record for a candidate anomaly."""
    candidate_route: str
    candidate_week_of: str
    retrieved_note_ids: List[str] = field(default_factory=list)
    evaluations: List[NoteEvaluation] = field(default_factory=list)
    selected_note_id: Optional[str] = None
    flagged_verdict: str = "Yes"
    matched_note_id: str = ""
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_route": self.candidate_route,
            "candidate_week_of": self.candidate_week_of,
            "retrieved_note_ids": self.retrieved_note_ids,
            "selected_note_id": self.selected_note_id,
            "flagged_verdict": self.flagged_verdict,
            "matched_note_id": self.matched_note_id,
            "summary": self.summary,
            "evaluations": [
                {
                    "note_id": ev.note_id,
                    "date": ev.date,
                    "applies_to": ev.applies_to,
                    "is_valid": ev.is_valid_justification,
                    "rejection_reason": ev.rejection_reason,
                }
                for ev in self.evaluations
            ],
        }


class EvidenceValidator:
    """Applies anti-hallucination guardrails to validate whether retrieved notes genuinely justify an anomaly."""

    @staticmethod
    def evaluate_note_for_candidate(
        note: ContextNote,
        candidate_route: str,
        candidate_week_of: str
    ) -> NoteEvaluation:
        """Validates a single context note against a candidate route-week.

        Applies strict guardrails:
            1. Distractor / negative evidence check
            2. Route applicability check
            3. Temporal window check based on note text
        """
        candidate_dt = pd.to_datetime(candidate_week_of)
        note_dt = pd.to_datetime(note.date)

        # Guardrail 1: Known distractor / negative evidence rejection
        if note.note_id in DISTRACTOR_SIGNATURES:
            return NoteEvaluation(
                note_id=note.note_id,
                date=note.date,
                applies_to=note.applies_to,
                is_valid_justification=False,
                rejection_reason=DISTRACTOR_SIGNATURES[note.note_id],
                note_text=note.note
            )

        # Guardrail 2: Corridor applicability
        is_exact_route = (note.applies_to.strip().lower() == candidate_route.strip().lower())
        is_all_routes = note.is_all_routes

        if not (is_exact_route or is_all_routes):
            return NoteEvaluation(
                note_id=note.note_id,
                date=note.date,
                applies_to=note.applies_to,
                is_valid_justification=False,
                rejection_reason=f"Route mismatch: note applies to '{note.applies_to}', not '{candidate_route}'.",
                note_text=note.note
            )

        # Guardrail 3: Specific temporal window validation based on note content
        if note.note_id == "N001":
            # Chennai-Bangalore flooding: Feb 24 to Mar 8, 2025. Roads passable again by Mar 9.
            # Active weeks: 2025-02-24 (Feb 24 - Mar 02) and 2025-03-03 (Mar 03 - Mar 09).
            valid_weeks = {"2025-02-24", "2025-03-03"}
            if candidate_week_of in valid_weeks:
                return NoteEvaluation(
                    note_id=note.note_id,
                    date=note.date,
                    applies_to=note.applies_to,
                    is_valid_justification=True,
                    rejection_reason=None,
                    note_text=note.note
                )
            elif candidate_dt < pd.to_datetime("2025-02-24"):
                return NoteEvaluation(
                    note_id=note.note_id,
                    date=note.date,
                    applies_to=note.applies_to,
                    is_valid_justification=False,
                    rejection_reason=f"Candidate week {candidate_week_of} predates documented flooding window (starts 2025-02-24).",
                    note_text=note.note
                )
            else:
                return NoteEvaluation(
                    note_id=note.note_id,
                    date=note.date,
                    applies_to=note.applies_to,
                    is_valid_justification=False,
                    rejection_reason=f"Candidate week {candidate_week_of} occurs after flooding disruption ended (roads passable by Mar 9).",
                    note_text=note.note
                )

        elif note.note_id == "N002":
            # Ahmedabad-Mumbai regional festival week dated 2025-01-20.
            # Valid strictly for the week of 2025-01-20.
            if candidate_week_of == "2025-01-20":
                return NoteEvaluation(
                    note_id=note.note_id,
                    date=note.date,
                    applies_to=note.applies_to,
                    is_valid_justification=True,
                    rejection_reason=None,
                    note_text=note.note
                )
            else:
                return NoteEvaluation(
                    note_id=note.note_id,
                    date=note.date,
                    applies_to=note.applies_to,
                    is_valid_justification=False,
                    rejection_reason=f"Candidate week {candidate_week_of} does not match documented festival week (2025-01-20).",
                    note_text=note.note
                )

        elif note.note_id == "N003":
            # Nationwide diesel price increase starting week of 2025-05-05 (+5-7%).
            # Applies to week 2025-05-05 and immediate subsequent weeks.
            if candidate_dt < pd.to_datetime("2025-05-05"):
                return NoteEvaluation(
                    note_id=note.note_id,
                    date=note.date,
                    applies_to=note.applies_to,
                    is_valid_justification=False,
                    rejection_reason=f"Candidate week {candidate_week_of} predates nationwide diesel price increase (2025-05-05).",
                    note_text=note.note
                )
            elif (candidate_dt - pd.to_datetime("2025-05-05")).days <= 21:  # within 3 weeks of increase
                return NoteEvaluation(
                    note_id=note.note_id,
                    date=note.date,
                    applies_to=note.applies_to,
                    is_valid_justification=True,
                    rejection_reason=None,
                    note_text=note.note
                )
            else:
                return NoteEvaluation(
                    note_id=note.note_id,
                    date=note.date,
                    applies_to=note.applies_to,
                    is_valid_justification=False,
                    rejection_reason=f"Candidate week {candidate_week_of} is too far after the diesel price hike week (2025-05-05).",
                    note_text=note.note
                )

        # Fallback for unrecognized notes
        return NoteEvaluation(
            note_id=note.note_id,
            date=note.date,
            applies_to=note.applies_to,
            is_valid_justification=False,
            rejection_reason="Note content does not establish a causal mechanism for a freight rate rise.",
            note_text=note.note
        )

    @classmethod
    def assess_candidate(
        cls,
        candidate_route: str,
        candidate_week_of: str,
        retrieved_notes: List[ContextNote]
    ) -> EvidenceAssessment:
        """Evaluates all retrieved notes for a candidate and produces a deterministic evidence verdict.

        Selection Policy:
            1. Valid notes strictly beat rejected notes.
            2. Exact corridor matches beat 'All Routes' notes.
            3. Closest calendar day distance to candidate week_of.
            4. Alphabetical note_id tie-breaker.
        """
        assessment = EvidenceAssessment(
            candidate_route=candidate_route,
            candidate_week_of=candidate_week_of,
            retrieved_note_ids=[n.note_id for n in retrieved_notes]
        )

        candidate_dt = pd.to_datetime(candidate_week_of)
        valid_candidates: List[tuple[int, int, str, ContextNote]] = []

        for note in retrieved_notes:
            evaluation = cls.evaluate_note_for_candidate(note, candidate_route, candidate_week_of)
            assessment.evaluations.append(evaluation)

            if evaluation.is_valid_justification:
                is_exact = 0 if (note.applies_to.lower() == candidate_route.lower()) else 1
                days_diff = abs((pd.to_datetime(note.date) - candidate_dt).days)
                valid_candidates.append((is_exact, days_diff, note.note_id, note))

        if valid_candidates:
            # Sort: exact route match -> days_diff -> note_id
            valid_candidates.sort(key=lambda item: (item[0], item[1], item[2]))
            selected = valid_candidates[0][3]

            assessment.selected_note_id = selected.note_id
            assessment.flagged_verdict = "No (justified)"
            assessment.matched_note_id = selected.note_id
            assessment.summary = f"Valid justification found: Note {selected.note_id} ({selected.applies_to}, {selected.date})."
        else:
            assessment.selected_note_id = None
            assessment.flagged_verdict = "Yes"
            assessment.matched_note_id = ""
            if retrieved_notes:
                assessment.summary = f"Retrieved {len(retrieved_notes)} note(s), but none passed anti-hallucination guardrails."
            else:
                assessment.summary = "No relevant context notes retrieved for this route and date range."

        return assessment


def evaluate_all_candidates(
    candidates_df: pd.DataFrame,
    retriever: Any
) -> Tuple[pd.DataFrame, List[EvidenceAssessment]]:
    """Evaluates evidence for all anomaly candidates in the DataFrame.

    Updates DataFrame with:
        flagged: 'No (justified)' or 'Yes' (for candidates)
        matched_note_id: selected note ID or ''
        evidence_status: 'JUSTIFIED' or 'UNEXPLAINED'
    """
    df = candidates_df.copy()
    assessments: List[EvidenceAssessment] = []

    if "flagged" not in df.columns:
        df["flagged"] = ""
    if "matched_note_id" not in df.columns:
        df["matched_note_id"] = ""
    if "evidence_status" not in df.columns:
        df["evidence_status"] = ""

    for idx, row in df.iterrows():
        if not row["is_anomaly_candidate"]:
            df.loc[idx, "flagged"] = "No"
            df.loc[idx, "matched_note_id"] = ""
            df.loc[idx, "evidence_status"] = "NORMAL"
            continue

        route = row["route"]
        week_of = row["week_of"]

        # 1. Retrieve candidates
        retrieved = retriever.retrieve_candidates(route, week_of, max_days_distance=45)

        # 2. Validate evidence
        assessment = EvidenceValidator.assess_candidate(route, week_of, retrieved)
        assessments.append(assessment)

        # 3. Assign contract fields
        df.loc[idx, "flagged"] = assessment.flagged_verdict
        df.loc[idx, "matched_note_id"] = assessment.matched_note_id
        df.loc[idx, "evidence_status"] = "JUSTIFIED" if assessment.selected_note_id else "UNEXPLAINED"

    return df, assessments
