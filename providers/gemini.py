"""Gemini model provider implementation."""

import logging
import time
from typing import Optional

from google import genai
from google.genai import types

from utils.gemini_errors import (
    check_vision_support,
    estimate_tokens_fallback,
    extract_gemini_usage,
    is_gemini_error_retryable,
    process_gemini_image,
)
from utils.thinking_mode import calculate_thinking_budget, validate_thinking_mode_support

from .base import ModelCapabilities, ModelProvider, ModelResponse, ProviderType
from .gemini_core import GeminiModelRegistry

logger = logging.getLogger(__name__)


class GeminiModelProvider(ModelProvider):
    """Google Gemini model provider implementation."""

    # Use shared model registry for consistent specifications
    def get_model_configurations(self) -> dict[str, ModelCapabilities]:
        """Get model configurations from shared registry."""
        # Only include models that Direct Gemini API actually supports
        # (exclude models that are only available via Vertex AI)
        direct_api_models = [
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
        ]

        provider_overrides = {
            "gemini-2.0-flash": {
                "aliases": ["flash-2.0", "flash2"],
                "friendly_name_override": "Gemini (Flash 2.0)",
            },
            "gemini-2.0-flash-lite": {
                "aliases": ["flashlite", "flash-lite"],
                "friendly_name_override": "Gemini (Flash Lite 2.0)",
            },
            "gemini-2.5-flash": {
                "aliases": ["flash", "flash2.5"],
                "friendly_name_override": "Gemini (Flash 2.5)",
            },
            "gemini-2.5-pro": {
                "aliases": ["pro", "gemini pro", "gemini-pro"],
                "friendly_name_override": "Gemini (Pro 2.5)",
            },
        }
        return GeminiModelRegistry.create_provider_models(
            provider_type=ProviderType.GOOGLE, provider_overrides=provider_overrides, model_filter=direct_api_models
        )

    # For backward compatibility, create SUPPORTED_MODELS as a property
    @property
    def SUPPORTED_MODELS(self) -> dict[str, ModelCapabilities]:
        """Backward compatibility property."""
        if not hasattr(self, "_cached_models"):
            self._cached_models = self.get_model_configurations()
        return self._cached_models

    # Thinking mode logic moved to utils/thinking_mode.py

    def __init__(self, api_key: str, **kwargs):
        """Initialize Gemini provider with API key."""
        super().__init__(api_key, **kwargs)
        self._client = None
        self._token_counters = {}  # Cache for token counting

    @property
    def client(self):
        """Lazy initialization of Gemini client."""
        if self._client is None:
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def get_capabilities(self, model_name: str) -> ModelCapabilities:
        """Get capabilities for a specific Gemini model."""
        # Resolve shorthand
        resolved_name = self._resolve_model_name(model_name)

        if resolved_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported Gemini model: {model_name}")

        # Check if model is allowed by restrictions
        from utils.model_restrictions import get_restriction_service

        restriction_service = get_restriction_service()
        # IMPORTANT: Parameter order is (provider_type, model_name, original_name)
        # resolved_name is the canonical model name, model_name is the user input
        if not restriction_service.is_allowed(ProviderType.GOOGLE, resolved_name, model_name):
            raise ValueError(f"Gemini model '{resolved_name}' is not allowed by restriction policy.")

        # Return the ModelCapabilities object from shared registry
        model_configs = self.get_model_configurations()
        if resolved_name not in model_configs:
            raise ValueError(f"Unsupported Gemini model: {resolved_name}")
        return model_configs[resolved_name]

    def generate_content(
        self,
        prompt: str,
        model_name: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_output_tokens: Optional[int] = None,
        thinking_mode: str = "medium",
        images: Optional[list[str]] = None,
        **kwargs,
    ) -> ModelResponse:
        """Generate content using Gemini model."""
        # Validate parameters
        resolved_name = self._resolve_model_name(model_name)
        self.validate_parameters(model_name, temperature)

        # Prepare content parts (text and potentially images)
        parts = []

        # Add system and user prompts as text
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt

        parts.append({"text": full_prompt})

        # Add images if provided and model supports vision
        if images and self._supports_vision(resolved_name):
            for image_path in images:
                try:
                    image_part = self._process_image(image_path)
                    if image_part:
                        parts.append(image_part)
                except Exception as e:
                    logger.warning(f"Failed to process image {image_path}: {e}")
                    # Continue with other images and text
                    continue
        elif images and not self._supports_vision(resolved_name):
            logger.warning(f"Model {resolved_name} does not support images, ignoring {len(images)} image(s)")

        # Create contents structure
        contents = [{"parts": parts}]

        # Prepare generation config
        generation_config = types.GenerateContentConfig(
            temperature=temperature,
            candidate_count=1,
        )

        # Add max output tokens if specified
        if max_output_tokens:
            generation_config.max_output_tokens = max_output_tokens

        # Add thinking configuration for models that support it
        if validate_thinking_mode_support(resolved_name, thinking_mode):
            actual_thinking_budget = calculate_thinking_budget(resolved_name, thinking_mode)
            if actual_thinking_budget > 0:
                generation_config.thinking_config = types.ThinkingConfig(thinking_budget=actual_thinking_budget)

        # Retry logic with progressive delays
        max_retries = 4  # Total of 4 attempts
        retry_delays = [1, 3, 5, 8]  # Progressive delays: 1s, 3s, 5s, 8s

        last_exception = None

        for attempt in range(max_retries):
            try:
                # Generate content
                response = self.client.models.generate_content(
                    model=resolved_name,
                    contents=contents,
                    config=generation_config,
                )

                # Extract usage information if available
                usage = self._extract_usage(response)

                return ModelResponse(
                    content=response.text,
                    usage=usage,
                    model_name=resolved_name,
                    friendly_name="Gemini",
                    provider=ProviderType.GOOGLE,
                    metadata={
                        "thinking_mode": (
                            thinking_mode if validate_thinking_mode_support(resolved_name, thinking_mode) else None
                        ),
                        "finish_reason": (
                            getattr(response.candidates[0], "finish_reason", "STOP") if response.candidates else "STOP"
                        ),
                    },
                )

            except Exception as e:
                last_exception = e

                # Check if this is a retryable error using structured error codes
                is_retryable = self._is_error_retryable(e)

                # If this is the last attempt or not retryable, give up
                if attempt == max_retries - 1 or not is_retryable:
                    break

                # Get progressive delay
                delay = retry_delays[attempt]

                # Log retry attempt
                logger.warning(
                    f"Gemini API error for model {resolved_name}, attempt {attempt + 1}/{max_retries}: {str(e)}. Retrying in {delay}s..."
                )
                time.sleep(delay)

        # If we get here, all retries failed
        actual_attempts = attempt + 1  # Convert from 0-based index to human-readable count
        error_msg = f"Gemini API error for model {resolved_name} after {actual_attempts} attempt{'s' if actual_attempts > 1 else ''}: {str(last_exception)}"
        raise RuntimeError(error_msg) from last_exception

    def count_tokens(self, text: str, model_name: str) -> int:
        """Count tokens for the given text using Gemini's tokenizer."""
        self._resolve_model_name(model_name)

        # For now, use shared estimation utility
        # TODO: Use actual Gemini tokenizer when available in SDK
        return estimate_tokens_fallback(text)

    def get_provider_type(self) -> ProviderType:
        """Get the provider type."""
        return ProviderType.GOOGLE

    def validate_model_name(self, model_name: str) -> bool:
        """Validate if the model name is supported and allowed."""
        resolved_name = self._resolve_model_name(model_name)

        # First check if model is supported
        if resolved_name not in self.SUPPORTED_MODELS:
            return False

        # Then check if model is allowed by restrictions
        from utils.model_restrictions import get_restriction_service

        restriction_service = get_restriction_service()
        # IMPORTANT: Parameter order is (provider_type, model_name, original_name)
        # resolved_name is the canonical model name, model_name is the user input
        if not restriction_service.is_allowed(ProviderType.GOOGLE, resolved_name, model_name):
            logger.debug(f"Gemini model '{model_name}' -> '{resolved_name}' blocked by restrictions")
            return False

        return True

    def supports_thinking_mode(self, model_name: str) -> bool:
        """Check if the model supports extended thinking mode."""
        resolved_name = self._resolve_model_name(model_name)
        return validate_thinking_mode_support(resolved_name, "medium")  # Check with any valid mode

    def get_thinking_budget(self, model_name: str, thinking_mode: str) -> int:
        """Get actual thinking token budget for a model and thinking mode."""
        resolved_name = self._resolve_model_name(model_name)
        return calculate_thinking_budget(resolved_name, thinking_mode)

    def _extract_usage(self, response) -> dict[str, int]:
        """Extract token usage from Gemini response."""
        return extract_gemini_usage(response)

    def _supports_vision(self, model_name: str) -> bool:
        """Check if the model supports vision (image processing)."""
        return check_vision_support(model_name)

    def _is_error_retryable(self, error: Exception) -> bool:
        """Determine if an error should be retried based on structured error codes."""
        return is_gemini_error_retryable(error)

    def _process_image(self, image_path: str) -> Optional[dict]:
        """Process an image for Gemini API."""
        return process_gemini_image(image_path, self.validate_image)
