"""Shared thinking mode utilities for Gemini-based providers.

This module provides shared thinking mode calculation logic that was previously
duplicated between the Direct Gemini API provider and Vertex AI provider.
It extracts the ~60 lines of duplicated thinking mode logic into a reusable utility.
"""

import logging
from typing import Optional

from providers.gemini_core import GeminiModelRegistry

logger = logging.getLogger(__name__)


# Thinking mode configurations - percentages of model's max_thinking_tokens
# These percentages work across all models that support thinking
THINKING_BUDGETS = {
    "minimal": 0.005,  # 0.5% of max - minimal thinking for fast responses
    "low": 0.08,  # 8% of max - light reasoning tasks
    "medium": 0.33,  # 33% of max - balanced reasoning (default)
    "high": 0.67,  # 67% of max - complex analysis
    "max": 1.0,  # 100% of max - full thinking budget
}


def get_thinking_budget_percentage(thinking_mode: str) -> float:
    """Get the percentage of max thinking tokens for a thinking mode.

    Args:
        thinking_mode: The thinking mode ("minimal", "low", "medium", "high", "max")

    Returns:
        Float percentage (0.0 to 1.0) of max thinking tokens to use
    """
    return THINKING_BUDGETS.get(thinking_mode, THINKING_BUDGETS["medium"])


def calculate_thinking_budget(model_id: str, thinking_mode: str) -> int:
    """Calculate actual thinking token budget for a model and thinking mode.

    This function provides the core thinking budget calculation logic that was
    previously duplicated between providers.

    Args:
        model_id: Canonical model identifier (e.g., "gemini-2.5-pro")
        thinking_mode: The thinking mode ("minimal", "low", "medium", "high", "max")

    Returns:
        Actual thinking token budget as integer
    """
    # Get model specification from registry
    spec = GeminiModelRegistry.get_spec(model_id)
    if not spec:
        logger.debug(f"Unknown model {model_id}, returning 0 thinking tokens")
        return 0

    # Check if model supports thinking
    if not spec.supports_extended_thinking:
        logger.debug(f"Model {model_id} does not support thinking mode")
        return 0

    # Check if thinking mode is valid
    if thinking_mode not in THINKING_BUDGETS:
        logger.debug(f"Invalid thinking mode '{thinking_mode}', using medium")
        thinking_mode = "medium"

    # Calculate budget
    max_thinking_tokens = spec.max_thinking_tokens
    if max_thinking_tokens == 0:
        logger.debug(f"Model {model_id} has 0 max thinking tokens")
        return 0

    budget_percentage = THINKING_BUDGETS[thinking_mode]
    actual_budget = int(max_thinking_tokens * budget_percentage)

    logger.debug(
        f"Model {model_id} thinking budget: {actual_budget} tokens "
        f"({thinking_mode} = {budget_percentage:.1%} of {max_thinking_tokens})"
    )

    return actual_budget


def validate_thinking_mode_support(model_id: str, thinking_mode: Optional[str]) -> bool:
    """Validate if a model supports the requested thinking mode.

    Args:
        model_id: Canonical model identifier
        thinking_mode: The thinking mode to validate

    Returns:
        True if model supports thinking mode, False otherwise
    """
    if thinking_mode is None:
        return True  # No thinking mode requested, always valid

    spec = GeminiModelRegistry.get_spec(model_id)
    if not spec:
        return False

    return spec.supports_extended_thinking and thinking_mode in THINKING_BUDGETS


def get_model_max_thinking_tokens(model_id: str) -> int:
    """Get the maximum thinking tokens for a model.

    Args:
        model_id: Canonical model identifier

    Returns:
        Maximum thinking tokens for the model
    """
    spec = GeminiModelRegistry.get_spec(model_id)
    return spec.max_thinking_tokens if spec else 0


def list_thinking_modes() -> list[str]:
    """List all available thinking modes.

    Returns:
        List of thinking mode names
    """
    return list(THINKING_BUDGETS.keys())


def get_thinking_mode_description(thinking_mode: str) -> str:
    """Get a human-readable description of a thinking mode.

    Args:
        thinking_mode: The thinking mode to describe

    Returns:
        Human-readable description
    """
    descriptions = {
        "minimal": "0.5% of max tokens - minimal thinking for fast responses",
        "low": "8% of max tokens - light reasoning tasks",
        "medium": "33% of max tokens - balanced reasoning (default)",
        "high": "67% of max tokens - complex analysis",
        "max": "100% of max tokens - full thinking budget",
    }
    return descriptions.get(thinking_mode, "Unknown thinking mode")
