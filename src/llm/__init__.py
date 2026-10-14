"""LLM Explanation and Token/Cost Accounting Package."""

from src.llm.cost_tracker import CostTracker, CallMetric, redact_secrets
from src.llm.prompts import (
    SYSTEM_PROMPT,
    build_explanation_prompt,
    validate_structured_explanation,
)
from src.llm.explainer import (
    BaseLLMProvider,
    OfflineProvider,
    MockLLMProvider,
    OpenAICompatibleProvider,
    ExplanationGenerator,
    generate_deterministic_explanation,
    get_llm_provider,
)

__all__ = [
    "CostTracker",
    "CallMetric",
    "redact_secrets",
    "SYSTEM_PROMPT",
    "build_explanation_prompt",
    "validate_structured_explanation",
    "BaseLLMProvider",
    "OfflineProvider",
    "MockLLMProvider",
    "OpenAICompatibleProvider",
    "ExplanationGenerator",
    "generate_deterministic_explanation",
    "get_llm_provider",
]
