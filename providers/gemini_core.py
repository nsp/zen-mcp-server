"""Core Gemini model specifications and shared utilities.

This module provides the single source of truth for Gemini model specifications
and shared utilities that can be used by different providers (Direct API, Vertex AI).
"""

from dataclasses import dataclass
from typing import Optional

from .base import ModelCapabilities, ProviderType, create_temperature_constraint


@dataclass
class GeminiModelSpec:
    """Canonical Gemini model specification.

    This represents the base specification for a Gemini model that can be
    adapted for different providers (Direct API, Vertex AI) with provider-specific
    overrides for enterprise configurations, regional limits, etc.
    """

    model_id: str
    friendly_name: str
    base_context_window: int  # Base specification
    base_max_output_tokens: int
    supports_extended_thinking: bool
    max_thinking_tokens: int = 0  # Only for thinking-capable models
    supports_system_prompts: bool = True
    supports_streaming: bool = True
    supports_function_calling: bool = True
    supports_json_mode: bool = True
    supports_images: bool = True
    max_image_size_mb: float = 20.0
    supports_temperature: bool = True
    description: str = ""

    # Provider-specific overrides can be applied at runtime
    # For example, Vertex AI might have different image size limits
    # or context windows due to enterprise configurations


class GeminiModelRegistry:
    """Registry of canonical Gemini model specifications.

    This serves as the single source of truth for Gemini model specifications,
    eliminating duplication between providers while allowing for provider-specific
    customizations where needed (e.g., enterprise tuning, regional restrictions).
    """

    # Single source of truth for Gemini model specs
    # These are based on Google's official documentation and represent
    # the canonical capabilities of each model
    CANONICAL_SPECS: dict[str, GeminiModelSpec] = {
        # Gemini 2.0 Models
        "gemini-2.0-flash-001": GeminiModelSpec(
            model_id="gemini-2.0-flash-001",
            friendly_name="Gemini 2.0 Flash (001)",
            base_context_window=1_048_576,  # 1M tokens
            base_max_output_tokens=65_536,
            supports_extended_thinking=True,
            max_thinking_tokens=24576,
            supports_images=True,
            max_image_size_mb=20.0,
            description="Latest Gemini 2.0 Flash model with enhanced capabilities",
        ),
        "gemini-2.0-flash-lite-001": GeminiModelSpec(
            model_id="gemini-2.0-flash-lite-001",
            friendly_name="Gemini 2.0 Flash-Lite (001)",
            base_context_window=1_048_576,  # 1M tokens
            base_max_output_tokens=65_536,
            supports_extended_thinking=False,
            max_thinking_tokens=0,
            supports_images=False,  # Lite version doesn't support images
            max_image_size_mb=0.0,
            description="Lightweight Gemini 2.0 Flash model, text-only",
        ),
        # Gemini 2.5 Models
        "gemini-2.5-flash": GeminiModelSpec(
            model_id="gemini-2.5-flash",
            friendly_name="Gemini 2.5 Flash",
            base_context_window=1_048_576,  # 1M tokens
            base_max_output_tokens=65_536,
            supports_extended_thinking=True,
            max_thinking_tokens=24576,  # Flash 2.5 thinking budget limit
            supports_images=True,
            max_image_size_mb=20.0,
            description="Ultra-fast model - Quick analysis, simple queries, rapid iterations",
        ),
        "gemini-2.5-flash-lite-preview-06-17": GeminiModelSpec(
            model_id="gemini-2.5-flash-lite-preview-06-17",
            friendly_name="Gemini 2.5 Flash-Lite (Preview)",
            base_context_window=1_000_000,  # 1M context
            base_max_output_tokens=8_192,
            supports_extended_thinking=False,
            max_thinking_tokens=0,
            supports_images=True,
            max_image_size_mb=20.0,
            description="Preview version of cost-effective model for high throughput tasks",
        ),
        "gemini-2.5-pro": GeminiModelSpec(
            model_id="gemini-2.5-pro",
            friendly_name="Gemini 2.5 Pro",
            base_context_window=1_048_576,  # 1M tokens
            base_max_output_tokens=65_536,
            supports_extended_thinking=True,
            max_thinking_tokens=32768,  # Max thinking tokens for Pro model
            supports_images=True,
            max_image_size_mb=32.0,  # Higher limit for Pro model
            description="Deep reasoning + thinking mode - Complex problems, architecture, deep analysis",
        ),
        # Legacy 1.5 Models (for backward compatibility)
        "gemini-1.5-pro-002": GeminiModelSpec(
            model_id="gemini-1.5-pro-002",
            friendly_name="Gemini 1.5 Pro (Legacy)",
            base_context_window=2_097_152,  # 2M tokens for legacy 1.5 Pro
            base_max_output_tokens=8_192,
            supports_extended_thinking=False,
            max_thinking_tokens=0,
            supports_images=True,
            max_image_size_mb=20.0,
            description="Legacy Gemini 1.5 Pro model (2M context)",
        ),
        "gemini-1.5-flash-002": GeminiModelSpec(
            model_id="gemini-1.5-flash-002",
            friendly_name="Gemini 1.5 Flash (Legacy)",
            base_context_window=1_048_576,  # 1M context
            base_max_output_tokens=8_192,
            supports_extended_thinking=False,
            max_thinking_tokens=0,
            supports_images=True,
            max_image_size_mb=20.0,
            description="Legacy Gemini 1.5 Flash model (1M context)",
        ),
        # Legacy Pro Vision Model
        "gemini-pro-vision": GeminiModelSpec(
            model_id="gemini-pro-vision",
            friendly_name="Gemini Pro Vision (Legacy)",
            base_context_window=32_000,
            base_max_output_tokens=4_096,
            supports_extended_thinking=False,
            max_thinking_tokens=0,
            supports_images=True,
            max_image_size_mb=20.0,
            description="Legacy Gemini Pro model with vision capabilities",
        ),
        "gemini-pro": GeminiModelSpec(
            model_id="gemini-pro",
            friendly_name="Gemini Pro (Legacy)",
            base_context_window=32_000,
            base_max_output_tokens=4_096,
            supports_extended_thinking=False,
            max_thinking_tokens=0,
            supports_images=False,
            max_image_size_mb=0.0,
            description="Legacy Gemini Pro model, text-only",
        ),
        # Alias models for backward compatibility
        "gemini-2.0-flash": GeminiModelSpec(
            model_id="gemini-2.0-flash",
            friendly_name="Gemini 2.0 Flash",
            base_context_window=1_048_576,  # 1M tokens
            base_max_output_tokens=65_536,
            supports_extended_thinking=True,
            max_thinking_tokens=24576,
            supports_images=True,
            max_image_size_mb=20.0,
            description="Latest fast model with experimental thinking, supports audio/video input",
        ),
        "gemini-2.0-flash-lite": GeminiModelSpec(
            model_id="gemini-2.0-flash-lite",
            friendly_name="Gemini 2.0 Flash-Lite",
            base_context_window=1_048_576,  # 1M tokens
            base_max_output_tokens=65_536,
            supports_extended_thinking=False,
            max_thinking_tokens=0,
            supports_images=False,  # Lite version doesn't support images
            max_image_size_mb=0.0,
            description="Lightweight fast model, text-only",
        ),
        "gemini-2.5-flash-lite": GeminiModelSpec(
            model_id="gemini-2.5-flash-lite",
            friendly_name="Gemini 2.5 Flash-Lite",
            base_context_window=1_000_000,  # 1M context
            base_max_output_tokens=8_192,
            supports_extended_thinking=False,
            max_thinking_tokens=0,
            supports_images=True,
            max_image_size_mb=20.0,
            description="Most cost-effective model for high throughput tasks",
        ),
        "gemini-2.0-flash-exp": GeminiModelSpec(
            model_id="gemini-2.0-flash-exp",
            friendly_name="Gemini 2.0 Flash (Experimental)",
            base_context_window=1_048_576,  # 1M context
            base_max_output_tokens=65_536,
            supports_extended_thinking=True,
            max_thinking_tokens=24576,
            supports_images=True,
            max_image_size_mb=20.0,
            description="Experimental version with latest features",
        ),
    }

    @classmethod
    def get_spec(cls, model_id: str) -> Optional[GeminiModelSpec]:
        """Get canonical specification for a model."""
        return cls.CANONICAL_SPECS.get(model_id)

    @classmethod
    def create_capabilities(
        cls,
        model_id: str,
        provider_type: ProviderType,
        aliases: Optional[list] = None,
        context_window_override: Optional[int] = None,
        max_output_override: Optional[int] = None,
        max_thinking_override: Optional[int] = None,
        image_size_override: Optional[float] = None,
        friendly_name_override: Optional[str] = None,
        description_override: Optional[str] = None,
    ) -> Optional[ModelCapabilities]:
        """Create ModelCapabilities from canonical spec with provider-specific overrides.

        This allows providers to customize model specifications for their specific
        infrastructure while maintaining a single source of truth for base capabilities.

        Args:
            model_id: Canonical model identifier
            provider_type: Which provider this is for (GOOGLE, VERTEX_AI, etc.)
            aliases: Provider-specific aliases
            context_window_override: Provider-specific context window limit
            max_output_override: Provider-specific output token limit
            max_thinking_override: Provider-specific thinking token limit
            image_size_override: Provider-specific image size limit
            friendly_name_override: Custom friendly name for this provider
            description_override: Custom description for this provider
        """
        spec = cls.get_spec(model_id)
        if not spec:
            return None

        return ModelCapabilities(
            provider=provider_type,
            model_name=model_id,
            friendly_name=friendly_name_override or spec.friendly_name,
            context_window=context_window_override or spec.base_context_window,
            max_output_tokens=max_output_override or spec.base_max_output_tokens,
            supports_extended_thinking=spec.supports_extended_thinking,
            max_thinking_tokens=max_thinking_override or spec.max_thinking_tokens,
            supports_system_prompts=spec.supports_system_prompts,
            supports_streaming=spec.supports_streaming,
            supports_function_calling=spec.supports_function_calling,
            supports_json_mode=spec.supports_json_mode,
            supports_images=spec.supports_images,
            max_image_size_mb=image_size_override or spec.max_image_size_mb,
            supports_temperature=spec.supports_temperature,
            temperature_constraint=create_temperature_constraint("range"),
            description=description_override or spec.description,
            aliases=aliases or [],
        )

    @classmethod
    def create_provider_models(
        cls,
        provider_type: ProviderType,
        provider_overrides: Optional[dict[str, dict]] = None,
        model_filter: Optional[list] = None,
    ) -> dict[str, ModelCapabilities]:
        """Create a complete model dictionary for a provider.

        Args:
            provider_type: The provider type
            provider_overrides: Dict of model_id -> override dict
            model_filter: Optional list of model IDs to include (None = all)

        Returns:
            Dictionary mapping model_id to ModelCapabilities
        """
        models = {}
        overrides = provider_overrides or {}
        model_ids = model_filter or list(cls.CANONICAL_SPECS.keys())

        for model_id in model_ids:
            if model_id not in cls.CANONICAL_SPECS:
                continue

            model_overrides = overrides.get(model_id, {})
            capabilities = cls.create_capabilities(model_id=model_id, provider_type=provider_type, **model_overrides)

            if capabilities:
                models[model_id] = capabilities

        return models

    @classmethod
    def list_models(cls) -> list[str]:
        """List all available model IDs."""
        return list(cls.CANONICAL_SPECS.keys())

    @classmethod
    def list_thinking_models(cls) -> list[str]:
        """List models that support thinking mode."""
        return [model_id for model_id, spec in cls.CANONICAL_SPECS.items() if spec.supports_extended_thinking]


class GeminiContentProcessor:
    """Shared utilities for processing Gemini content across providers."""

    @staticmethod
    def prepare_text_content(prompt: str) -> str:
        """Prepare text content for Gemini API."""
        return prompt.strip()

    @staticmethod
    def validate_thinking_mode(model_id: str, thinking_mode: Optional[str]) -> bool:
        """Validate if model supports thinking mode."""
        spec = GeminiModelRegistry.get_spec(model_id)
        return spec and spec.supports_extended_thinking and thinking_mode is not None

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough token estimation for Gemini models."""
        # Simple estimation: ~4 characters per token
        return len(text) // 4

    @staticmethod
    def get_thinking_token_limit(model_id: str) -> int:
        """Get the thinking token limit for a model."""
        spec = GeminiModelRegistry.get_spec(model_id)
        return spec.max_thinking_tokens if spec else 0
