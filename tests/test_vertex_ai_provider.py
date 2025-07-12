"""Tests for the unified Vertex AI model provider"""

import os
from unittest.mock import Mock, patch

import pytest

from providers import ModelProviderRegistry
from providers.base import ProviderType
from providers.vertex_ai import VertexAIModelProvider


class TestUnifiedVertexAIProvider:
    """Test unified Vertex AI model provider"""

    def test_provider_initialization_without_project_id(self):
        """Test provider initialization fails without project ID"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="VERTEX_AI_PROJECT_ID environment variable is required"):
                VertexAIModelProvider()

    @patch("utils.credential_manager.default")
    @patch("providers.vertex_ai.vertexai")
    def test_provider_initialization_with_project_id(self, mock_vertexai, mock_default):
        """Test provider initialization with project ID"""
        # Mock credentials
        mock_credentials = Mock()
        mock_default.return_value = (mock_credentials, None)

        with patch.dict(os.environ, {"VERTEX_AI_PROJECT_ID": "test-project"}):
            provider = VertexAIModelProvider()

            assert provider._project_id == "test-project"
            assert provider._location == "us-central1"  # Default location
            assert provider.get_provider_type() == ProviderType.VERTEX_AI

            # Verify credentials were obtained
            mock_default.assert_called_once()

    @patch("utils.credential_manager.default")
    @patch("providers.vertex_ai.vertexai")
    def test_provider_initialization_with_custom_location(self, mock_vertexai, mock_default):
        """Test provider initialization with custom location"""
        mock_credentials = Mock()
        mock_default.return_value = (mock_credentials, None)

        with patch.dict(os.environ, {"VERTEX_AI_PROJECT_ID": "test-project", "VERTEX_AI_LOCATION": "europe-west4"}):
            provider = VertexAIModelProvider()

            assert provider._location == "europe-west4"

    @patch("utils.credential_manager.default")
    @patch("providers.vertex_ai.vertexai")
    def test_get_capabilities_gemini_models(self, mock_vertexai, mock_default):
        """Test getting capabilities for Gemini models"""
        mock_credentials = Mock()
        mock_default.return_value = (mock_credentials, None)

        with patch.dict(os.environ, {"VERTEX_AI_PROJECT_ID": "test-project"}):
            provider = VertexAIModelProvider()

            # Test supported Gemini model
            capabilities = provider.get_capabilities("gemini-1.5-pro-002")
            assert capabilities.provider == ProviderType.VERTEX_AI
            assert capabilities.model_name == "gemini-1.5-pro-002"
            assert capabilities.context_window == 2_097_152
            assert capabilities.supports_images is True

            # Test Gemini alias
            capabilities = provider.get_capabilities("vertex-pro")
            assert capabilities.model_name == "gemini-1.5-pro-002"

    @patch("utils.credential_manager.default")
    @patch("providers.vertex_ai.vertexai")
    def test_get_capabilities_claude_models(self, mock_vertexai, mock_default):
        """Test getting capabilities for Claude models"""
        mock_credentials = Mock()
        mock_default.return_value = (mock_credentials, None)

        with patch.dict(os.environ, {"VERTEX_AI_PROJECT_ID": "test-project"}):
            provider = VertexAIModelProvider()

            # Test supported Claude model
            capabilities = provider.get_capabilities("claude-sonnet-4@20250514")
            assert capabilities.provider == ProviderType.VERTEX_AI
            assert capabilities.model_name == "claude-sonnet-4@20250514"
            assert capabilities.context_window == 200_000
            assert capabilities.supports_images is True

            # Test Claude alias
            capabilities = provider.get_capabilities("sonnet-4")
            assert capabilities.model_name == "claude-sonnet-4@20250514"

    @patch("utils.credential_manager.default")
    @patch("providers.vertex_ai.vertexai")
    def test_validate_model_name(self, mock_vertexai, mock_default):
        """Test model name validation"""
        mock_credentials = Mock()
        mock_default.return_value = (mock_credentials, None)

        with patch.dict(os.environ, {"VERTEX_AI_PROJECT_ID": "test-project"}):
            provider = VertexAIModelProvider()

            # Valid Gemini models
            assert provider.validate_model_name("gemini-1.5-pro-002") is True
            assert provider.validate_model_name("vertex-pro") is True  # Alias

            # Valid Claude models
            assert provider.validate_model_name("claude-sonnet-4@20250514") is True
            assert provider.validate_model_name("sonnet-4") is True  # Alias

            # Invalid model
            assert provider.validate_model_name("unknown-model") is False

    @patch("utils.credential_manager.default")
    @patch("providers.vertex_ai.vertexai")
    def test_supports_thinking_mode(self, mock_vertexai, mock_default):
        """Test thinking mode support check"""
        mock_credentials = Mock()
        mock_default.return_value = (mock_credentials, None)

        with patch.dict(os.environ, {"VERTEX_AI_PROJECT_ID": "test-project"}):
            provider = VertexAIModelProvider()

            # Gemini models
            assert provider.supports_thinking_mode("gemini-1.5-pro-002") is False
            assert provider.supports_thinking_mode("gemini-2.0-flash-exp") is True

            # Claude models don't support thinking mode
            assert provider.supports_thinking_mode("claude-sonnet-4@20250514") is False

    @patch("utils.credential_manager.default")
    @patch("providers.vertex_ai.vertexai")
    @patch("providers.vertex_ai.GenerativeModel")
    def test_generate_gemini_content_success(self, mock_model_class, mock_vertexai, mock_default):
        """Test successful content generation with Gemini models"""
        # Setup mocks
        mock_credentials = Mock()
        mock_default.return_value = (mock_credentials, None)

        # Mock model instance
        mock_model = Mock()
        mock_model_class.return_value = mock_model

        # Mock response
        mock_response = Mock()
        mock_response.text = "Generated content"
        mock_response.usage_metadata = Mock(prompt_token_count=100, candidates_token_count=50, total_token_count=150)
        mock_model.generate_content.return_value = mock_response

        with patch.dict(os.environ, {"VERTEX_AI_PROJECT_ID": "test-project"}):
            provider = VertexAIModelProvider()

            response = provider.generate_content(
                prompt="Test prompt", model_name="gemini-1.5-pro-002", temperature=0.7, max_output_tokens=1000
            )

            # Ensure vertexai.init was called during generate_content
            mock_vertexai.init.assert_called_once_with(
                project="test-project", location="us-central1", credentials=mock_credentials
            )

            assert response.content == "Generated content"
            assert response.usage["input_tokens"] == 100
            assert response.usage["output_tokens"] == 50
            assert response.model_name == "gemini-1.5-pro-002"
            assert response.provider == ProviderType.VERTEX_AI

    @patch("utils.credential_manager.default")
    def test_generate_claude_content_success(self, mock_default):
        """Test successful content generation with Claude models"""
        # Setup mocks
        mock_credentials = Mock()
        mock_credentials.token = "test-token"
        mock_default.return_value = (mock_credentials, None)

        # Mock HTTP response with proper attributes
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "test response text"
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": "Generated content from Claude"}],
            "usage": {"input_tokens": 100, "output_tokens": 50},
            "model": "claude-sonnet-4@20250514",
            "stop_reason": "end_turn",
        }

        # Configure the mock client instance
        mock_client_instance = Mock()
        mock_client_instance.post.return_value = mock_response

        with patch.dict(os.environ, {"VERTEX_AI_PROJECT_ID": "test-project", "VERTEX_AI_LOCATION": "us-east5"}):
            provider = VertexAIModelProvider()

            # Patch the _claude_client directly on the provider instance
            provider._claude_client = mock_client_instance

            response = provider.generate_content(
                prompt="Test prompt", model_name="claude-sonnet-4@20250514", temperature=0.7, max_output_tokens=1000
            )

            assert response.content == "Generated content from Claude"
            assert response.usage["input_tokens"] == 100
            assert response.usage["output_tokens"] == 50
            assert response.model_name == "claude-sonnet-4@20250514"
            assert response.provider == ProviderType.VERTEX_AI

    @patch("utils.credential_manager.default")
    @patch("providers.vertex_ai.vertexai")
    def test_count_tokens_for_different_models(self, mock_vertexai, mock_default):
        """Test token counting for both Gemini and Claude models"""
        mock_credentials = Mock()
        mock_default.return_value = (mock_credentials, None)

        with patch.dict(os.environ, {"VERTEX_AI_PROJECT_ID": "test-project"}):
            provider = VertexAIModelProvider()

            # Claude models use simple estimation
            text = "This is a test sentence with some words."
            tokens = provider.count_tokens(text, "claude-sonnet-4@20250514")
            assert tokens == len(text) // 4

            # Gemini models would try to use API but fall back to estimation
            tokens = provider.count_tokens(text, "gemini-1.5-pro-002")
            assert tokens == len(text) // 4

    @patch.dict(os.environ, {"VERTEX_AI_PROJECT_ID": "test-project"})
    @patch("utils.credential_manager.default")
    @patch("providers.vertex_ai.vertexai")
    @pytest.mark.no_mock_provider
    def test_registry_integration(self, mock_vertexai, mock_default):
        """Test integration with provider registry"""
        mock_credentials = Mock()
        mock_default.return_value = (mock_credentials, None)

        # Register the provider
        ModelProviderRegistry.register_provider(ProviderType.VERTEX_AI, VertexAIModelProvider)

        # Get provider from registry
        provider = ModelProviderRegistry.get_provider(ProviderType.VERTEX_AI)

        assert provider is not None
        assert isinstance(provider, VertexAIModelProvider)

        # Test getting provider for Gemini model
        provider = ModelProviderRegistry.get_provider_for_model("gemini-1.5-pro-002")
        if provider:
            assert isinstance(provider, VertexAIModelProvider)

        # Test getting provider for Claude model
        provider = ModelProviderRegistry.get_provider_for_model("claude-sonnet-4@20250514")
        if provider:
            assert isinstance(provider, VertexAIModelProvider)

    @patch("utils.credential_manager.default")
    @patch("providers.vertex_ai.vertexai")
    def test_model_routing(self, mock_vertexai, mock_default):
        """Test that models are correctly routed to appropriate implementations"""
        mock_credentials = Mock()
        mock_credentials.token = "test-token"
        mock_default.return_value = (mock_credentials, None)

        with patch.dict(os.environ, {"VERTEX_AI_PROJECT_ID": "test-project"}):
            provider = VertexAIModelProvider()

            # Test _is_claude_model method
            assert provider._is_claude_model("claude-sonnet-4@20250514") is True
            assert provider._is_claude_model("sonnet-4") is True  # Alias
            assert provider._is_claude_model("gemini-1.5-pro-002") is False
            assert provider._is_claude_model("vertex-pro") is False  # Gemini alias

            # Test _resolve_model_name method
            assert provider._resolve_model_name("sonnet-4") == "claude-sonnet-4@20250514"
            assert provider._resolve_model_name("vertex-pro") == "gemini-1.5-pro-002"
            assert provider._resolve_model_name("unknown-model") == "unknown-model"

    @patch("utils.credential_manager.default")
    @patch("providers.vertex_ai.vertexai")
    def test_error_handling(self, mock_vertexai, mock_default):
        """Test error handling for various scenarios"""
        mock_credentials = Mock()
        mock_default.return_value = (mock_credentials, None)

        with patch.dict(os.environ, {"VERTEX_AI_PROJECT_ID": "test-project"}):
            provider = VertexAIModelProvider()

            # Test retryable vs non-retryable errors
            from google.api_core import exceptions as google_exceptions

            # Authentication error - not retryable
            auth_error = google_exceptions.Unauthenticated("Not authenticated")
            assert provider._is_error_retryable(auth_error) is False

            # Service unavailable - retryable
            service_error = google_exceptions.ServiceUnavailable("Service down")
            assert provider._is_error_retryable(service_error) is True

            # Rate limit - sometimes retryable
            rate_error = google_exceptions.ResourceExhausted("Too many requests")
            assert provider._is_error_retryable(rate_error) is True
