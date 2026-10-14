"""Token usage accounting and USD cost tracking module."""

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
from src.config import MODEL_PRICING_RATES, RUN_METRICS_FILE

logger = logging.getLogger(__name__)


def redact_secrets(text: str) -> str:
    """Redacts potential API keys and secrets from logs and metric dumps."""
    if not isinstance(text, str):
        return str(text)
    # Mask common key patterns (e.g. sk-..., AIza..., bearer tokens)
    redacted = re.sub(r"(sk-[a-zA-Z0-9_-]{8,})", r"sk-***REDACTED***", text)
    redacted = re.sub(r"(AIza[a-zA-Z0-9_-]{8,})", r"AIza***REDACTED***", redacted)
    redacted = re.sub(r"(Bearer\s+)[a-zA-Z0-9_\-\.]{8,}", r"\1***REDACTED***", redacted)
    return redacted


@dataclass
class CallMetric:
    """Record of a single model inference attempt."""
    call_id: int
    timestamp: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    is_estimated: bool
    estimated_cost_usd: float
    status: str  # "SUCCESS", "FALLBACK", "ERROR"
    error_message: Optional[str] = None


class CostTracker:
    """Tracks token counts, LLM API calls, errors, fallbacks, and estimated USD spend."""

    def __init__(self, pricing_rates: Optional[Dict[str, Dict[str, float]]] = None):
        self.pricing_rates = pricing_rates or MODEL_PRICING_RATES
        self.calls: List[CallMetric] = []
        self._call_counter = 0

    def estimate_tokens_from_text(self, text: str) -> int:
        """Lightweight heuristic tokenizer estimate (~4 characters per token) when provider metadata is absent."""
        if not text:
            return 0
        return max(1, len(text) // 4)

    def calculate_cost(self, provider_or_model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculates estimated USD cost based on configured pricing rates."""
        rates = self.pricing_rates.get(provider_or_model.lower())
        if not rates:
            # Fallback to default/offline rates
            rates = self.pricing_rates.get("offline", {"input_per_m": 0.0, "output_per_m": 0.0})

        input_cost = (input_tokens / 1_000_000.0) * rates.get("input_per_m", 0.0)
        output_cost = (output_tokens / 1_000_000.0) * rates.get("output_per_m", 0.0)
        return round(input_cost + output_cost, 6)

    def record_call(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        is_estimated: bool = False,
        status: str = "SUCCESS",
        error_message: Optional[str] = None
    ) -> CallMetric:
        """Records an actual or estimated LLM invocation."""
        self._call_counter += 1
        cost = self.calculate_cost(model, input_tokens, output_tokens)
        if cost == 0.0:
            cost = self.calculate_cost(provider, input_tokens, output_tokens)

        metric = CallMetric(
            call_id=self._call_counter,
            timestamp=datetime.now(timezone.utc).isoformat(),
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            is_estimated=is_estimated,
            estimated_cost_usd=cost,
            status=status,
            error_message=redact_secrets(error_message) if error_message else None
        )
        self.calls.append(metric)
        return metric

    def get_summary(self) -> Dict[str, Any]:
        """Produces aggregated audit metrics across all tracked calls."""
        total_calls = len(self.calls)
        successful_calls = sum(1 for c in self.calls if c.status == "SUCCESS")
        fallback_calls = sum(1 for c in self.calls if c.status == "FALLBACK")
        error_calls = sum(1 for c in self.calls if c.status == "ERROR")

        total_input_tokens = sum(c.input_tokens for c in self.calls)
        total_output_tokens = sum(c.output_tokens for c in self.calls)
        total_tokens = total_input_tokens + total_output_tokens

        actual_token_calls = sum(1 for c in self.calls if not c.is_estimated and c.total_tokens > 0)
        estimated_token_calls = sum(1 for c in self.calls if c.is_estimated and c.total_tokens > 0)

        total_cost_usd = round(sum(c.estimated_cost_usd for c in self.calls), 6)

        return {
            "total_calls": total_calls,
            "successful_calls": successful_calls,
            "fallback_calls": fallback_calls,
            "error_calls": error_calls,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_tokens,
            "actual_metadata_calls": actual_token_calls,
            "estimated_metadata_calls": estimated_token_calls,
            "total_cost_usd": total_cost_usd,
        }

    def save_run_metrics(self, output_path: Path | str = RUN_METRICS_FILE) -> Path:
        """Persists the complete audit log to run_metrics.json."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        summary = self.get_summary()
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": summary,
            "calls": [asdict(c) for c in self.calls]
        }

        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        logger.info("Saved run metrics and token cost log to %s", out)
        return out
