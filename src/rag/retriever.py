"""Context Notes Retriever Module.

Loads real context notes from context_notes.csv and performs transparent,
deterministic retrieval based on corridor applicability and temporal proximity.
Does NOT decide whether a retrieved note justifies an anomaly.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import pandas as pd
from src.config import CONTEXT_NOTES_FILE

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContextNote:
    """Represents an official context note from context_notes.csv."""
    note_id: str
    date: str  # YYYY-MM-DD
    applies_to: str
    note: str

    @property
    def is_all_routes(self) -> bool:
        return self.applies_to.strip().lower() == "all routes"


class ContextNoteRetriever:
    """Loads and retrieves relevant context notes for candidate route-weeks."""

    def __init__(self, notes_file: Path | str = CONTEXT_NOTES_FILE):
        self.notes_file = Path(notes_file)
        self.notes: List[ContextNote] = []
        self._load_notes()

    def _load_notes(self) -> None:
        if not self.notes_file.is_file():
            raise FileNotFoundError(f"Context notes file not found at: {self.notes_file}")

        df = pd.read_csv(self.notes_file)
        required_cols = ["note_id", "date", "applies_to", "note"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column in context_notes.csv: {col}")

        self.notes = [
            ContextNote(
                note_id=str(row["note_id"]).strip(),
                date=str(row["date"]).strip(),
                applies_to=str(row["applies_to"]).strip(),
                note=str(row["note"]).strip()
            )
            for _, row in df.iterrows()
        ]
        logger.info("Loaded %d context notes from %s", len(self.notes), self.notes_file)

    def retrieve_candidates(
        self,
        route: str,
        week_of: str,
        max_days_distance: int = 45
    ) -> List[ContextNote]:
        """Retrieves candidate context notes matching the corridor or marked 'All Routes'.

        Filters by route applicability and temporal proximity.
        Sorting is deterministic: exact route matches first, then sorted by absolute
        calendar day distance to the candidate week_of, then by note_id.

        Args:
            route: Candidate route string (e.g. 'Ahmedabad-Mumbai').
            week_of: Candidate Monday date string (e.g. '2025-01-20').
            max_days_distance: Maximum absolute calendar days allowed between note date and week_of.

        Returns:
            List of candidate ContextNote objects.
        """
        candidate_dt = pd.to_datetime(week_of)
        retrieved: List[tuple[int, int, ContextNote]] = []

        for note in self.notes:
            note_dt = pd.to_datetime(note.date)
            days_diff = abs((note_dt - candidate_dt).days)

            # Route match: either exact corridor or "All Routes"
            is_exact_route = (note.applies_to.lower() == route.lower())
            is_all_routes = note.is_all_routes

            if not (is_exact_route or is_all_routes):
                continue

            if days_diff <= max_days_distance:
                # Rank priority: 0 for exact route match, 1 for All Routes
                route_priority = 0 if is_exact_route else 1
                retrieved.append((route_priority, days_diff, note))

        # Sort: priority -> days_diff -> note_id
        retrieved.sort(key=lambda item: (item[0], item[1], item[2].note_id))

        results = [item[2] for item in retrieved]
        logger.debug(
            "Retrieved %d candidate notes for %s on %s (within %d days)",
            len(results),
            route,
            week_of,
            max_days_distance
        )
        return results

    def get_closest_note(self, route: str, week_of: str) -> Optional[ContextNote]:
        """Finds the single closest note temporally (route-matched or All Routes)."""
        all_matches = self.retrieve_candidates(route, week_of, max_days_distance=365)
        return all_matches[0] if all_matches else None
