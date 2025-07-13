"""Comprehensive tests for shared Gemini components.

These tests validate the shared components (gemini_core, thinking_mode, gemini_errors)
that eliminate code duplication between Gemini and Vertex AI providers.
They ensure the shared components work correctly for Phase 1 (Gemini provider)
and will continue to work unchanged for Phase 2 (Vertex AI provider refactoring).
"""

from unittest.mock import Mock

from providers.base import ProviderType
from providers.gemini_core import GeminiModelRegistry
from utils.gemini_errors import (
    check_vision_support,
    estimate_tokens_fallback,
    extract_gemini_usage,
    is_gemini_error_retryable,
    process_gemini_image,
)
from utils.thinking_mode import (
    THINKING_BUDGETS,
    calculate_thinking_budget,
    validate_thinking_mode_support,
)


class TestGeminiModelRegistry:
    """Test the shared Gemini model specifications registry."""

    def test_get_spec_valid_models(self):
        """Test getting specifications for valid models."""
        # Test standard Gemini models
        spec = GeminiModelRegistry.get_spec("gemini-2.5-flash")
        assert spec is not None
        assert spec.model_id == "gemini-2.5-flash"
        assert spec.base_context_window == 1_048_576
        assert spec.supports_images is True
        assert spec.supports_extended_thinking is True

        spec = GeminiModelRegistry.get_spec("gemini-2.5-pro")
        assert spec is not None
        assert spec.model_id == "gemini-2.5-pro"
        assert spec.base_context_window == 1_048_576
        assert spec.supports_images is True
        assert spec.supports_extended_thinking is True
        assert spec.max_thinking_tokens == 32768

    def test_get_spec_invalid_model(self):
        """Test getting specification for invalid model returns None."""
        spec = GeminiModelRegistry.get_spec("unknown-model")
        assert spec is None

    def test_list_models(self):
        """Test getting list of all model IDs."""
        model_ids = GeminiModelRegistry.list_models()
        assert isinstance(model_ids, list)
        assert len(model_ids) > 0

        # Verify key models are present
        assert "gemini-2.5-flash" in model_ids
        assert "gemini-2.5-pro" in model_ids
        assert "gemini-1.5-pro-002" in model_ids

    def test_list_thinking_models(self):
        """Test getting list of thinking-capable models."""
        thinking_models = GeminiModelRegistry.list_thinking_models()
        assert isinstance(thinking_models, list)

        # Known thinking models
        assert "gemini-2.5-flash" in thinking_models
        assert "gemini-2.5-pro" in thinking_models
        assert "gemini-2.0-flash" in thinking_models

        # Known non-thinking models should not be present
        assert "gemini-1.5-pro-002" not in thinking_models

    def test_create_provider_models_basic(self):
        """Test creating provider models with basic configuration."""
        provider_models = GeminiModelRegistry.create_provider_models(provider_type=ProviderType.GOOGLE)

        assert isinstance(provider_models, dict)
        assert len(provider_models) > 0

        # Check a known model
        assert "gemini-2.5-flash" in provider_models
        capabilities = provider_models["gemini-2.5-flash"]
        assert capabilities.provider == ProviderType.GOOGLE
        assert capabilities.model_name == "gemini-2.5-flash"
        assert capabilities.context_window == 1_048_576

    def test_create_provider_models_with_overrides(self):
        """Test creating provider models with provider-specific overrides."""
        provider_overrides = {
            "gemini-2.5-flash": {
                "aliases": ["flash", "flash2.5"],
                "friendly_name_override": "Gemini (Flash 2.5)",
            },
            "gemini-2.5-pro": {
                "aliases": ["pro", "gemini-pro"],
                "friendly_name_override": "Gemini (Pro 2.5)",
            },
        }

        provider_models = GeminiModelRegistry.create_provider_models(
            provider_type=ProviderType.VERTEX_AI, provider_overrides=provider_overrides
        )

        # Check that overrides are applied
        flash_cap = provider_models["gemini-2.5-flash"]
        assert flash_cap.aliases == ["flash", "flash2.5"]
        assert flash_cap.friendly_name == "Gemini (Flash 2.5)"
        assert flash_cap.provider == ProviderType.VERTEX_AI

    def test_create_provider_models_with_filter(self):
        """Test creating provider models with model filtering."""
        model_filter = ["gemini-2.5-flash", "gemini-2.5-pro"]

        provider_models = GeminiModelRegistry.create_provider_models(
            provider_type=ProviderType.GOOGLE, model_filter=model_filter
        )

        # Should only contain filtered models
        assert len(provider_models) == 2
        assert "gemini-2.5-flash" in provider_models
        assert "gemini-2.5-pro" in provider_models
        assert "gemini-1.5-pro-002" not in provider_models

    def test_provider_model_consistency(self):
        """Test that provider models maintain consistency with base specs."""
        provider_models = GeminiModelRegistry.create_provider_models(provider_type=ProviderType.GOOGLE)

        for model_id, capabilities in provider_models.items():
            base_spec = GeminiModelRegistry.get_spec(model_id)
            assert base_spec is not None

            # Core properties should match base spec
            assert capabilities.context_window == base_spec.base_context_window
            assert capabilities.supports_images == base_spec.supports_images
            assert capabilities.max_output_tokens == base_spec.base_max_output_tokens


class TestThinkingModeUtils:
    """Test the shared thinking mode utilities."""

    def test_thinking_budgets_constants(self):
        """Test that thinking budget constants are properly defined."""
        assert isinstance(THINKING_BUDGETS, dict)

        # Check required thinking modes
        required_modes = ["minimal", "low", "medium", "high", "max"]
        for mode in required_modes:
            assert mode in THINKING_BUDGETS
            assert isinstance(THINKING_BUDGETS[mode], (int, float))
            assert 0 <= THINKING_BUDGETS[mode] <= 1

        # Check that values are reasonable
        assert THINKING_BUDGETS["minimal"] < THINKING_BUDGETS["low"]
        assert THINKING_BUDGETS["low"] < THINKING_BUDGETS["medium"]
        assert THINKING_BUDGETS["medium"] < THINKING_BUDGETS["high"]
        assert THINKING_BUDGETS["high"] < THINKING_BUDGETS["max"]

    def test_validate_thinking_mode_support_valid_models(self):
        """Test thinking mode validation for models that support it."""
        # Models that support thinking mode
        thinking_models = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"]

        for model in thinking_models:
            assert validate_thinking_mode_support(model, "medium") is True
            assert validate_thinking_mode_support(model, "high") is True
            assert validate_thinking_mode_support(model, "max") is True

    def test_validate_thinking_mode_support_invalid_models(self):
        """Test thinking mode validation for models that don't support it."""
        # Models that don't support thinking mode
        non_thinking_models = ["gemini-1.5-pro-002", "gemini-1.5-flash-002"]

        for model in non_thinking_models:
            assert validate_thinking_mode_support(model, "medium") is False
            assert validate_thinking_mode_support(model, "high") is False

    def test_validate_thinking_mode_support_unknown_model(self):
        """Test thinking mode validation for unknown models."""
        assert validate_thinking_mode_support("unknown-model", "medium") is False

    def test_calculate_thinking_budget_valid_models(self):
        """Test thinking budget calculation for models that support it."""
        model = "gemini-2.5-pro"

        # Test different thinking modes
        budget = calculate_thinking_budget(model, "medium")
        assert budget > 0

        low_budget = calculate_thinking_budget(model, "low")
        high_budget = calculate_thinking_budget(model, "high")
        max_budget = calculate_thinking_budget(model, "max")

        assert low_budget < budget < high_budget < max_budget

    def test_calculate_thinking_budget_invalid_models(self):
        """Test thinking budget calculation for models that don't support it."""
        # Model that doesn't support thinking mode
        budget = calculate_thinking_budget("gemini-1.5-pro-002", "medium")
        assert budget == 0

        # Unknown model
        budget = calculate_thinking_budget("unknown-model", "medium")
        assert budget == 0

    def test_calculate_thinking_budget_math(self):
        """Test that thinking budget calculations are mathematically correct."""
        model = "gemini-2.5-pro"
        spec = GeminiModelRegistry.get_spec(model)
        max_tokens = spec.max_thinking_tokens

        for mode, percentage in THINKING_BUDGETS.items():
            budget = calculate_thinking_budget(model, mode)
            expected = int(max_tokens * percentage)
            assert budget == expected


class TestGeminiErrorHandling:
    """Test the shared Gemini error handling utilities."""

    def test_is_gemini_error_retryable_rate_limits(self):
        """Test retry logic for rate limiting errors."""
        # Retryable rate limit errors
        retryable_errors = [
            Exception("429 Too Many Requests"),
            Exception("503 Service Unavailable"),
            Exception("timeout occurred"),
        ]

        for error in retryable_errors:
            assert is_gemini_error_retryable(error) is True

    def test_is_gemini_error_retryable_non_retryable_limits(self):
        """Test retry logic for non-retryable limit errors."""
        # Non-retryable limit errors
        non_retryable_errors = [
            Exception("429 quota exceeded for billing account"),
            Exception("resource exhausted: context length exceeded"),
            Exception("token limit exceeded for request"),
            Exception("request too large for model"),
        ]

        for error in non_retryable_errors:
            assert is_gemini_error_retryable(error) is False

    def test_is_gemini_error_retryable_network_errors(self):
        """Test retry logic for network-related errors."""
        # Retryable network errors
        retryable_errors = [
            Exception("Connection timeout"),
            Exception("Network unreachable"),
            Exception("503 Service Unavailable"),
            Exception("502 Bad Gateway"),
            Exception("SSL handshake failed"),
        ]

        for error in retryable_errors:
            assert is_gemini_error_retryable(error) is True

    def test_is_gemini_error_retryable_permanent_errors(self):
        """Test retry logic for permanent errors."""
        # Non-retryable permanent errors
        non_retryable_errors = [
            Exception("401 Unauthorized"),
            Exception("403 Forbidden"),
            Exception("400 Bad Request"),
            Exception("404 Not Found"),
        ]

        for error in non_retryable_errors:
            assert is_gemini_error_retryable(error) is False

    def test_is_gemini_error_retryable_structured_errors(self):
        """Test retry logic for structured errors with details."""
        # Mock error with structured details
        structured_error = Mock()
        structured_error.__str__ = Mock(return_value="429 Too Many Requests")
        structured_error.details = "quota exceeded for user"

        # Should be non-retryable due to details
        assert is_gemini_error_retryable(structured_error) is False

        # Mock error with retryable details
        retryable_error = Mock()
        retryable_error.__str__ = Mock(return_value="429 Too Many Requests")
        retryable_error.details = "temporary rate limiting"

        # Should be retryable
        assert is_gemini_error_retryable(retryable_error) is True

    def test_check_vision_support(self):
        """Test vision support checking for different models."""
        # Models with vision support
        vision_models = [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.0-flash",
            "gemini-1.5-pro-002",
            "gemini-1.5-flash-002",
        ]

        for model in vision_models:
            assert check_vision_support(model) is True

        # Models without vision support (hypothetical)
        assert check_vision_support("gemini-text-only") is False
        assert check_vision_support("unknown-model") is False

    def test_process_gemini_image_file_path(self):
        """Test image processing with file paths."""
        # Mock validation function
        mock_validate = Mock()
        mock_validate.return_value = (b"fake_image_bytes", "image/png")

        result = process_gemini_image("/path/to/image.png", mock_validate)

        assert result is not None
        assert "inline_data" in result
        assert result["inline_data"]["mime_type"] == "image/png"
        assert "data" in result["inline_data"]

        # Verify validation function was called
        mock_validate.assert_called_once_with("/path/to/image.png")

    def test_process_gemini_image_data_url(self):
        """Test image processing with data URLs."""
        # Mock validation function
        mock_validate = Mock()
        mock_validate.return_value = (b"fake_bytes", "image/jpeg")

        data_url = "data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="

        result = process_gemini_image(data_url, mock_validate)

        assert result is not None
        assert "inline_data" in result
        assert result["inline_data"]["mime_type"] == "image/jpeg"
        assert (
            result["inline_data"]["data"]
            == "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        )

    def test_process_gemini_image_validation_error(self):
        """Test image processing with validation errors."""
        # Mock validation function that raises error
        mock_validate = Mock()
        mock_validate.side_effect = ValueError("Invalid image format")

        result = process_gemini_image("/invalid/path.txt", mock_validate)

        assert result is None

    def test_extract_gemini_usage_complete(self):
        """Test usage extraction with complete metadata."""
        # Mock response with full usage metadata
        mock_response = Mock()
        mock_metadata = Mock()
        mock_metadata.prompt_token_count = 100
        mock_metadata.candidates_token_count = 50
        mock_response.usage_metadata = mock_metadata

        usage = extract_gemini_usage(mock_response)

        assert usage["input_tokens"] == 100
        assert usage["output_tokens"] == 50
        assert usage["total_tokens"] == 150

    def test_extract_gemini_usage_partial(self):
        """Test usage extraction with partial metadata."""
        # Mock response with only input tokens
        mock_response = Mock()
        mock_metadata = Mock()
        mock_metadata.prompt_token_count = 100
        mock_metadata.candidates_token_count = None
        mock_response.usage_metadata = mock_metadata

        usage = extract_gemini_usage(mock_response)

        assert usage["input_tokens"] == 100
        assert "output_tokens" not in usage
        assert "total_tokens" not in usage

    def test_extract_gemini_usage_no_metadata(self):
        """Test usage extraction with no metadata."""
        # Mock response without usage metadata
        mock_response = Mock()
        del mock_response.usage_metadata  # Ensure attribute doesn't exist

        usage = extract_gemini_usage(mock_response)

        assert usage == {}

    def test_estimate_tokens_fallback(self):
        """Test fallback token estimation."""
        test_cases = [
            ("Hello world", 2),  # 11 chars / 4 = 2
            ("This is a longer sentence for testing.", 9),  # 39 chars / 4 = 9
            ("", 0),  # Empty string
            ("a", 0),  # Single char rounds down to 0
            ("test", 1),  # Exactly 4 chars = 1 token
        ]

        for text, expected_tokens in test_cases:
            actual_tokens = estimate_tokens_fallback(text)
            assert actual_tokens == expected_tokens


class TestSharedComponentsIntegration:
    """Test integration between shared components."""

    def test_registry_and_thinking_mode_integration(self):
        """Test that registry and thinking mode utilities work together."""
        # Get a model that supports thinking mode
        spec = GeminiModelRegistry.get_spec("gemini-2.5-pro")
        assert spec is not None
        assert spec.supports_extended_thinking is True

        # Test thinking mode validation uses registry
        assert validate_thinking_mode_support("gemini-2.5-pro", "medium") is True

        # Test budget calculation uses registry
        budget = calculate_thinking_budget("gemini-2.5-pro", "medium")
        expected = int(spec.max_thinking_tokens * THINKING_BUDGETS["medium"])
        assert budget == expected

    def test_registry_and_vision_support_integration(self):
        """Test that registry and vision support work together."""
        # Test models that should support vision
        vision_models = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.5-pro-002"]

        for model in vision_models:
            spec = GeminiModelRegistry.get_spec(model)
            registry_supports_vision = spec.supports_images if spec else False
            utils_supports_vision = check_vision_support(model)

            # Both should agree on vision support
            assert registry_supports_vision == utils_supports_vision

    def test_provider_models_maintain_thinking_compatibility(self):
        """Test that provider models maintain thinking mode compatibility."""
        # Create provider models for Direct Gemini API
        provider_models = GeminiModelRegistry.create_provider_models(provider_type=ProviderType.GOOGLE)

        # Test that thinking mode utilities work with provider models
        for model_id in provider_models:
            spec = GeminiModelRegistry.get_spec(model_id)
            if spec and spec.supports_extended_thinking:
                # Should validate successfully
                assert validate_thinking_mode_support(model_id, "medium") is True

                # Should calculate budget correctly
                budget = calculate_thinking_budget(model_id, "high")
                assert budget > 0
            else:
                # Should not support thinking mode
                assert validate_thinking_mode_support(model_id, "medium") is False
                assert calculate_thinking_budget(model_id, "medium") == 0

    def test_phase_2_compatibility_simulation(self):
        """Test that shared components will work for Phase 2 (Vertex AI)."""
        # Simulate how Vertex AI provider would use shared components
        vertex_ai_overrides = {
            "gemini-2.5-pro": {
                "aliases": ["vertex-pro", "vertex-gemini-pro"],
                "friendly_name_override": "Vertex AI (Gemini Pro 2.5)",
            },
            "gemini-2.5-flash": {
                "aliases": ["vertex-flash", "vertex-gemini-flash"],
                "friendly_name_override": "Vertex AI (Gemini Flash 2.5)",
            },
        }

        # Create Vertex AI provider models
        vertex_models = GeminiModelRegistry.create_provider_models(
            provider_type=ProviderType.VERTEX_AI, provider_overrides=vertex_ai_overrides
        )

        # Test that all shared utilities work with Vertex AI models
        for model_id in vertex_models:
            capabilities = vertex_models[model_id]

            # Registry integration
            spec = GeminiModelRegistry.get_spec(model_id)
            assert spec is not None
            assert capabilities.context_window == spec.base_context_window

            # Thinking mode integration
            if spec.supports_extended_thinking:
                assert validate_thinking_mode_support(model_id, "medium") is True
                budget = calculate_thinking_budget(model_id, "medium")
                assert budget > 0

            # Vision support integration
            assert capabilities.supports_images == check_vision_support(model_id)

            # Error handling (would work the same)
            fake_error = Exception("429 Rate limit exceeded")
            assert is_gemini_error_retryable(fake_error) is True
