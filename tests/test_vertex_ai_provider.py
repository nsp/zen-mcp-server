"""Tests for the unified Vertex AI model provider"""

import os
from unittest.mock import Mock, patch

import pytest

from providers import ModelProviderRegistry
from providers.base import ProviderType

# Check if vertex AI dependencies are available
try:
    from providers.vertex_ai import VertexAIModelProvider

    vertex_ai_available = True
except ImportError:
    VertexAIModelProvider = None
    vertex_ai_available = False

pytestmark = pytest.mark.skipif(
    not vertex_ai_available,
    reason="vertexai package not available - install with 'pip install google-cloud-aiplatform'",
)


# Test data matrices for parameterized testing
PROVIDER_INIT_SCENARIOS = [
    # (env_vars, expected_project, expected_location, should_succeed)
    ({}, None, None, False),  # No project ID
    ({"VERTEX_AI_PROJECT_ID": "test-project"}, "test-project", "us-central1", True),
    (
        {"VERTEX_AI_PROJECT_ID": "test-project", "VERTEX_AI_LOCATION": "europe-west4"},
        "test-project",
        "europe-west4",
        True,
    ),
]

MODEL_VALIDATION_MATRIX = [
    ("gemini-1.5-pro-002", True),
    ("vertex-pro", True),  # Alias
    ("claude-sonnet-4@20250514", True),
    ("vertex-sonnet-4", True),  # Alias
    ("unknown-model", False),
]

THINKING_MODE_MATRIX = [
    ("gemini-1.5-pro-002", False),
    ("gemini-2.5-pro", True),
    ("claude-sonnet-4@20250514", False),
]

MODEL_CAPABILITIES_MATRIX = [
    # (model_name, expected_context, expected_supports_images, is_claude)
    ("gemini-1.5-pro-002", 2_097_152, True, False),
    ("claude-sonnet-4@20250514", 200_000, True, True),
    ("vertex-pro", 1_048_576, True, False),  # Alias for gemini-2.5-pro
    ("vertex-sonnet-4", 200_000, True, True),  # Alias for claude-sonnet-4
]

MODEL_ROUTING_MATRIX = [
    # (input_model, expected_resolved, is_claude)
    ("vertex-sonnet-4", "claude-sonnet-4@20250514", True),
    ("vertex-pro", "gemini-2.5-pro", False),
    ("unknown-model", "unknown-model", False),
]


@pytest.fixture
def mock_vertex_provider():
    """Fixture for creating mocked Vertex AI provider instances."""
    with patch("google.auth.default") as mock_auth, patch("google.cloud.aiplatform.init"):

        mock_credentials = Mock()
        mock_credentials.universe_domain = "googleapis.com"
        mock_auth.return_value = (mock_credentials, "test-project")

        def _create_provider(env_vars=None):
            env_vars = env_vars or {"VERTEX_AI_PROJECT_ID": "test-project"}
            with patch.dict(os.environ, env_vars, clear=True):
                return VertexAIModelProvider()

        yield _create_provider


@pytest.mark.xfail(reason="Vertex AI tests require authentication credentials")
class TestUnifiedVertexAIProvider:
    """Test unified Vertex AI model provider"""

    @pytest.mark.parametrize("env_vars,expected_project,expected_location,should_succeed", PROVIDER_INIT_SCENARIOS)
    def test_provider_initialization(self, env_vars, expected_project, expected_location, should_succeed):
        """Test provider initialization with various environment configurations."""
        with patch("google.auth.default") as mock_auth, patch("google.cloud.aiplatform.init"):

            mock_credentials = Mock()
            mock_credentials.universe_domain = "googleapis.com"
            mock_auth.return_value = (mock_credentials, expected_project)

            with patch.dict(os.environ, env_vars, clear=True):
                if should_succeed:
                    provider = VertexAIModelProvider()
                    assert provider.project_id == expected_project
                    assert provider.location == expected_location
                    assert provider.get_provider_type() == ProviderType.VERTEX_AI
                else:
                    with pytest.raises(ValueError, match="VERTEX_AI_PROJECT_ID required"):
                        VertexAIModelProvider()

    @pytest.mark.parametrize("model_name,expected_context,expected_images,is_claude", MODEL_CAPABILITIES_MATRIX)
    def test_model_capabilities(self, mock_vertex_provider, model_name, expected_context, expected_images, is_claude):
        """Test getting capabilities for all model types."""
        provider = mock_vertex_provider()

        capabilities = provider.get_capabilities(model_name)
        assert capabilities.provider == ProviderType.VERTEX_AI
        assert capabilities.context_window == expected_context
        assert capabilities.supports_images == expected_images

        # Verify model routing is correct
        assert provider._is_claude_model(model_name) == is_claude

    @pytest.mark.parametrize("model_name,expected_valid", MODEL_VALIDATION_MATRIX)
    def test_validate_model_name(self, mock_vertex_provider, model_name, expected_valid):
        """Test model name validation."""
        provider = mock_vertex_provider()
        assert provider.validate_model_name(model_name) == expected_valid

    @pytest.mark.parametrize("model_name,expected_support", THINKING_MODE_MATRIX)
    def test_supports_thinking_mode(self, mock_vertex_provider, model_name, expected_support):
        """Test thinking mode support check."""
        provider = mock_vertex_provider()
        assert provider.supports_thinking_mode(model_name) == expected_support

    @patch("vertexai.generative_models.GenerativeModel")
    def test_generate_gemini_content_success(self, mock_model_class, mock_vertex_provider):
        """Test successful content generation with Gemini models."""
        # Mock model instance and response
        mock_model = Mock()
        mock_model_class.return_value = mock_model

        mock_response = Mock()
        mock_response.text = "Generated content"
        mock_response.usage_metadata = Mock(prompt_token_count=100, candidates_token_count=50)
        mock_model.generate_content.return_value = mock_response

        provider = mock_vertex_provider()
        response = provider.generate_content(
            prompt="Test prompt", model_name="gemini-2.5-flash", temperature=0.7, max_output_tokens=1000
        )

        assert response.content == "Generated content"
        assert response.usage["input_tokens"] == 100
        assert response.usage["output_tokens"] == 50
        assert response.model_name == "gemini-2.5-flash"
        assert response.provider == ProviderType.VERTEX_AI

    @patch("google.cloud.aiplatform.Model")
    def test_generate_claude_content_success(self, mock_model_class, mock_vertex_provider):
        """Test successful content generation with Claude models."""
        # Mock Claude model and response
        mock_model = Mock()
        mock_model_class.return_value = mock_model

        mock_prediction = {
            "content": [{"type": "text", "text": "Generated content from Claude"}],
            "usage": {"input_tokens": 100, "output_tokens": 50},
        }
        mock_response = Mock()
        mock_response.predictions = [mock_prediction]
        mock_model.predict.return_value = mock_response

        provider = mock_vertex_provider()
        response = provider.generate_content(
            prompt="Test prompt", model_name="claude-sonnet-4@20250514", temperature=0.7, max_output_tokens=1000
        )

        assert response.content == "Generated content from Claude"
        assert response.usage["input_tokens"] == 100
        assert response.usage["output_tokens"] == 50
        assert response.model_name == "claude-sonnet-4@20250514"
        assert response.provider == ProviderType.VERTEX_AI

    def test_count_tokens_for_different_models(self, mock_vertex_provider):
        """Test token counting for both Gemini and Claude models."""
        provider = mock_vertex_provider()
        text = "This is a test sentence with some words."

        # Claude models use simple estimation
        claude_tokens = provider.count_tokens(text, "claude-sonnet-4@20250514")
        assert claude_tokens == len(text) // 4

        # Gemini models try API but fall back to estimation in test
        gemini_tokens = provider.count_tokens(text, "gemini-2.5-flash")
        assert gemini_tokens == len(text) // 4

    @pytest.mark.no_mock_provider
    def test_registry_integration(self, mock_vertex_provider):
        """Test integration with provider registry."""
        # Register the provider
        ModelProviderRegistry.register_provider(ProviderType.VERTEX_AI, VertexAIModelProvider)

        # Get provider from registry
        provider = ModelProviderRegistry.get_provider(ProviderType.VERTEX_AI)
        assert provider is not None
        assert isinstance(provider, VertexAIModelProvider)

        # Test getting provider for models
        for model_name in ["gemini-2.5-flash", "claude-sonnet-4@20250514"]:
            provider = ModelProviderRegistry.get_provider_for_model(model_name)
            if provider:
                assert isinstance(provider, VertexAIModelProvider)

    @pytest.mark.parametrize("input_model,expected_resolved,is_claude", MODEL_ROUTING_MATRIX)
    def test_model_routing(self, mock_vertex_provider, input_model, expected_resolved, is_claude):
        """Test that models are correctly routed to appropriate implementations."""
        provider = mock_vertex_provider()

        assert provider._resolve_model_name(input_model) == expected_resolved
        assert provider._is_claude_model(input_model) == is_claude
