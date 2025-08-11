"""
Pytest configuration for Zen MCP Server tests

This module provides centralized test configuration, fixtures, and mock helpers
that can be reused across all test modules. It eliminates code duplication
and provides consistent test patterns.
"""

import asyncio
import base64
import importlib
import os
import sys
from pathlib import Path
from typing import Any, Callable, Optional
from unittest.mock import Mock, patch

import pytest

# Ensure the parent directory is in the Python path for imports
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))


# Set default model to a specific value for tests to avoid auto mode
# This prevents all tests from failing due to missing model parameter
os.environ["DEFAULT_MODEL"] = "gemini-2.5-flash"

# Force reload of config module to pick up the env var
import config  # noqa: E402

importlib.reload(config)

# Note: This creates a test sandbox environment
# Tests create their own temporary directories as needed

from providers.base import ModelProvider, ModelResponse, ProviderType  # noqa: E402

# Sample test data
SAMPLE_IMAGE_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
SAMPLE_IMAGE_BYTES = base64.b64decode(SAMPLE_IMAGE_BASE64)
LARGE_TEST_DATA_SIZE = 21 * 1024 * 1024  # 21MB


# Provider test configuration matrices
PROVIDER_TEST_CONFIGS = {
    ProviderType.GOOGLE: {
        "models": ["gemini-2.5-flash", "gemini-2.5-pro"],
        "required_env": ["GEMINI_API_KEY"],
        "supports_thinking": True,
        "supports_images": True,
    },
    ProviderType.OPENAI: {
        "models": ["gpt-4o-mini", "gpt-4o"],
        "required_env": ["OPENAI_API_KEY"],
        "supports_thinking": False,
        "supports_images": True,
    },
    ProviderType.XAI: {
        "models": ["grok-2-vision-1212", "grok-2-1212"],
        "required_env": ["XAI_API_KEY"],
        "supports_thinking": False,
        "supports_images": True,
    },
}

MODEL_TEST_SCENARIOS = [
    ("gemini-2.5-flash", ProviderType.GOOGLE, True, True),
    ("gemini-2.5-pro", ProviderType.GOOGLE, True, True),
    ("gpt-4o-mini", ProviderType.OPENAI, False, True),
    ("gpt-4o", ProviderType.OPENAI, False, True),
    ("grok-2-vision-1212", ProviderType.XAI, False, True),
]

THINKING_MODE_SCENARIOS = [
    ("gemini-2.5-flash", True),
    ("gemini-2.5-pro", True),
    ("gpt-4o-mini", False),
    ("grok-2-vision-1212", False),
]

ERROR_HANDLING_SCENARIOS = [
    ("rate_limit", "429", True),
    ("auth_error", "401", False),
    ("not_found", "404", False),
    ("timeout", "timeout", True),
    ("server_error", "503", True),
]


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def project_path(tmp_path):
    """Temporary directory fixture for file operation tests."""
    return tmp_path


@pytest.fixture
def mock_provider_factory():
    """Factory fixture for creating mock providers with various configurations."""

    def _create_provider(
        provider_type: ProviderType = ProviderType.GOOGLE,
        api_key: str = "test-key",
        supports_thinking: bool = True,
        supports_images: bool = True,
        **kwargs,
    ) -> Mock:
        """Create a mock provider with specified capabilities."""
        provider = Mock(spec=ModelProvider)
        provider.get_provider_type.return_value = provider_type
        provider.supports_thinking_mode.return_value = supports_thinking
        provider.validate_model_name.return_value = True

        # Mock capabilities
        capabilities = Mock()
        capabilities.supports_images = supports_images
        capabilities.supports_thinking = supports_thinking
        capabilities.context_window = 100000
        capabilities.max_output_tokens = 8192
        provider.get_capabilities.return_value = capabilities

        # Set additional attributes
        for key, value in kwargs.items():
            setattr(provider, key, value)

        return provider

    return _create_provider


def _set_dummy_keys_if_missing():
    """Set dummy API keys only when they are completely absent."""
    for var in ("GEMINI_API_KEY", "OPENAI_API_KEY", "XAI_API_KEY"):
        if not os.environ.get(var):
            os.environ[var] = "dummy-key-for-tests"


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line("markers", "asyncio: mark test as async")
    config.addinivalue_line("markers", "no_mock_provider: disable automatic provider mocking")
    # Assume we need dummy keys until we learn otherwise
    config._needs_dummy_keys = True


def pytest_collection_modifyitems(session, config, items):
    """Hook that runs after test collection to check for no_mock_provider markers."""
    # Always set dummy keys if real keys are missing
    # This ensures tests work in CI even with no_mock_provider marker
    _set_dummy_keys_if_missing()


@pytest.fixture(autouse=True)
def mock_provider_availability(request, monkeypatch):
    """
    Automatically mock provider availability for all tests to prevent
    effective auto mode from being triggered when DEFAULT_MODEL is unavailable.

    This fixture ensures that when tests run with dummy API keys,
    the tools don't require model selection unless explicitly testing auto mode.
    """
    # Skip this fixture for tests that need real providers
    if hasattr(request, "node"):
        marker = request.node.get_closest_marker("no_mock_provider")
        if marker:
            return

    # Ensure providers are registered (in case other tests cleared the registry)
    from providers.base import ProviderType
    from providers.gemini import GeminiModelProvider
    from providers.openai_provider import OpenAIModelProvider
    from providers.registry import ModelProviderRegistry
    from providers.xai import XAIModelProvider

    registry = ModelProviderRegistry()

    if ProviderType.GOOGLE not in registry._providers:
        ModelProviderRegistry.register_provider(ProviderType.GOOGLE, GeminiModelProvider)
    if ProviderType.OPENAI not in registry._providers:
        ModelProviderRegistry.register_provider(ProviderType.OPENAI, OpenAIModelProvider)
    if ProviderType.XAI not in registry._providers:
        ModelProviderRegistry.register_provider(ProviderType.XAI, XAIModelProvider)

    # Ensure CUSTOM provider is registered if needed for integration tests
    if (
        os.getenv("CUSTOM_API_URL")
        and "test_prompt_regression.py" in os.getenv("PYTEST_CURRENT_TEST", "")
        and ProviderType.CUSTOM not in registry._providers
    ):
        from providers.custom import CustomProvider

        def custom_provider_factory(api_key=None):
            base_url = os.getenv("CUSTOM_API_URL", "")
            return CustomProvider(api_key=api_key or "", base_url=base_url)

        ModelProviderRegistry.register_provider(ProviderType.CUSTOM, custom_provider_factory)

    from unittest.mock import MagicMock

    original_get_provider = ModelProviderRegistry.get_provider_for_model

    def mock_get_provider_for_model(model_name):
        # If it's a test looking for unavailable models, return None
        if model_name in ["unavailable-model", "gpt-5-turbo", "o3"]:
            return None
        # For common test models, return a mock provider
        if model_name in ["gemini-2.5-flash", "gemini-2.5-pro", "pro", "flash", "local-llama"]:
            # Try to use the real provider first if it exists
            real_provider = original_get_provider(model_name)
            if real_provider:
                return real_provider

            # Otherwise create a mock
            provider = MagicMock()
            # Set up the model capabilities mock with actual values
            capabilities = MagicMock()
            if model_name == "local-llama":
                capabilities.context_window = 128000  # 128K tokens for local-llama
                capabilities.supports_extended_thinking = False
                capabilities.input_cost_per_1k = 0.0  # Free local model
                capabilities.output_cost_per_1k = 0.0  # Free local model
            else:
                capabilities.context_window = 1000000  # 1M tokens for Gemini models
                capabilities.supports_extended_thinking = False
                capabilities.input_cost_per_1k = 0.075
                capabilities.output_cost_per_1k = 0.3
            provider.get_model_capabilities.return_value = capabilities
            return provider
        # Otherwise use the original logic
        return original_get_provider(model_name)

    monkeypatch.setattr(ModelProviderRegistry, "get_provider_for_model", mock_get_provider_for_model)

    # Also mock is_effective_auto_mode for all BaseTool instances to return False
    # unless we're specifically testing auto mode behavior
    from tools.shared.base_tool import BaseTool

    def mock_is_effective_auto_mode(self):
        # If this is an auto mode test file or specific auto mode test, use the real logic
        test_file = request.node.fspath.basename if hasattr(request, "node") and hasattr(request.node, "fspath") else ""
        test_name = request.node.name if hasattr(request, "node") else ""

        # Allow auto mode for tests in auto mode files or with auto in the name
        if (
            "auto_mode" in test_file.lower()
            or "auto" in test_name.lower()
            or "intelligent_fallback" in test_file.lower()
            or "per_tool_model_defaults" in test_file.lower()
        ):
            # Call original method logic
            from config import DEFAULT_MODEL

            if DEFAULT_MODEL.lower() == "auto":
                return True
            provider = ModelProviderRegistry.get_provider_for_model(DEFAULT_MODEL)
            return provider is None
        # For all other tests, return False to disable auto mode
        return False

    monkeypatch.setattr(BaseTool, "is_effective_auto_mode", mock_is_effective_auto_mode)


@pytest.fixture
def mock_response_factory():
    """Factory fixture for creating mock responses with various configurations."""

    def _create_response(
        content: str = "Test response",
        model_name: str = "test-model",
        provider: ProviderType = ProviderType.GOOGLE,
        input_tokens: int = 100,
        output_tokens: int = 50,
        **kwargs,
    ) -> Mock:
        """Create a mock response with specified characteristics."""
        response = Mock(spec=ModelResponse)
        response.content = content
        response.model_name = model_name
        response.provider = provider
        response.usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

        # Set additional attributes
        for key, value in kwargs.items():
            setattr(response, key, value)

        return response

    return _create_response


@pytest.fixture
def mock_error_factory():
    """Factory fixture for creating mock errors with various types."""

    def _create_error(
        error_type: str = "api_error",
        message: str = "Test error",
        status_code: Optional[int] = None,
        retryable: bool = False,
        **kwargs,
    ) -> Exception:
        """Create a mock error with specified characteristics."""
        if error_type == "rate_limit":
            error = Exception(f"429 {message}")
        elif error_type == "timeout":
            error = Exception(f"timeout {message}")
        elif error_type == "auth":
            error = Exception(f"401 {message}")
        elif error_type == "not_found":
            error = Exception(f"404 {message}")
        else:
            error = Exception(message)

        # Add custom attributes for testing
        error.status_code = status_code
        error.retryable = retryable

        for key, value in kwargs.items():
            setattr(error, key, value)

        return error

    return _create_error


@pytest.fixture
def patch_factory():
    """Factory fixture for creating context manager patches."""
    active_patches = []

    def _create_patch(target: str, **kwargs) -> patch:
        """Create a patch context manager."""
        p = patch(target, **kwargs)
        active_patches.append(p)
        return p

    yield _create_patch

    # Cleanup active patches
    for p in active_patches:
        try:
            if p.is_local:
                p.stop()
        except Exception:
            pass  # Ignore cleanup errors


class MockHelpers:
    """Centralized mock helper utilities."""

    @staticmethod
    def create_mock_model_capabilities(
        model_name: str = "test-model",
        provider: ProviderType = ProviderType.GOOGLE,
        context_window: int = 100000,
        max_output_tokens: int = 8192,
        supports_thinking: bool = True,
        supports_images: bool = True,
        **kwargs,
    ) -> Mock:
        """Create mock ModelCapabilities object."""
        from providers.base import ModelCapabilities

        capabilities = Mock(spec=ModelCapabilities)
        capabilities.model_name = model_name
        capabilities.provider = provider
        capabilities.context_window = context_window
        capabilities.max_output_tokens = max_output_tokens
        capabilities.supports_thinking = supports_thinking
        capabilities.supports_images = supports_images

        for key, value in kwargs.items():
            setattr(capabilities, key, value)

        return capabilities

    @staticmethod
    def create_mock_validation_result(
        data: bytes = SAMPLE_IMAGE_BYTES,
        mime_type: str = "image/png",
        should_raise: bool = False,
        error_message: str = "Validation error",
    ) -> Callable:
        """Create mock validation function."""

        def mock_validate(image_path: str, max_size_mb: Optional[float] = None):
            if should_raise:
                raise ValueError(error_message)
            return data, mime_type

        return mock_validate

    @staticmethod
    def create_mock_gemini_response(
        text: str = "Generated content", input_tokens: int = 100, output_tokens: int = 50, **kwargs
    ) -> Mock:
        """Create mock Gemini API response."""
        response = Mock()
        response.text = text

        # Mock usage metadata
        usage_metadata = Mock()
        usage_metadata.prompt_token_count = input_tokens
        usage_metadata.candidates_token_count = output_tokens
        response.usage_metadata = usage_metadata

        for key, value in kwargs.items():
            setattr(response, key, value)

        return response

    @staticmethod
    def create_mock_vertex_response(
        content: str = "Generated content", input_tokens: int = 100, output_tokens: int = 50, **kwargs
    ) -> Mock:
        """Create mock Vertex AI response."""
        prediction = {
            "content": [{"type": "text", "text": content}],
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        }

        response = Mock()
        response.predictions = [prediction]

        for key, value in kwargs.items():
            setattr(response, key, value)

        return response

    @staticmethod
    def setup_environment_variables(env_vars: dict[str, str]) -> patch:
        """Set up environment variables for testing."""
        return patch.dict(os.environ, env_vars, clear=False)

    @staticmethod
    def assert_model_response_format(response: Any) -> None:
        """Assert that a response has the expected model response format."""
        assert hasattr(response, "content")
        assert hasattr(response, "usage")
        assert hasattr(response, "model_name")
        assert hasattr(response, "provider")

        # Check usage format
        usage = response.usage
        assert isinstance(usage, dict)
        assert "input_tokens" in usage or "output_tokens" in usage

    @staticmethod
    def assert_error_characteristics(
        error: Exception,
        expected_type: type = Exception,
        should_contain: Optional[str] = None,
        should_be_retryable: Optional[bool] = None,
    ) -> None:
        """Assert error characteristics."""

        assert isinstance(error, expected_type)

        if should_contain:
            assert should_contain in str(error)

        if should_be_retryable is not None:
            # Check if error has retryable attribute or use common patterns
            if hasattr(error, "retryable"):
                assert error.retryable == should_be_retryable
            else:
                # Use common patterns to determine retryability
                error_str = str(error).lower()
                is_retryable = any(
                    pattern in error_str for pattern in ["429", "503", "502", "timeout", "connection", "network"]
                )
                non_retryable = any(
                    pattern in error_str
                    for pattern in ["quota exceeded", "context length", "token limit", "401", "403", "400"]
                )

                if should_be_retryable:
                    assert is_retryable and not non_retryable
                else:
                    assert not is_retryable or non_retryable


@pytest.fixture
def mock_helpers():
    """Fixture providing access to MockHelpers utilities."""
    return MockHelpers


# Helper functions for common test patterns
def skip_if_no_api_key(provider_name: str, env_var: str):
    """Skip test if API key environment variable is not set."""
    return pytest.mark.skipif(not os.getenv(env_var), reason=f"{provider_name} API key not available in {env_var}")


def parametrize_models(models: list[str]):
    """Create parametrize decorator for testing multiple models."""
    return pytest.mark.parametrize("model_name", models)


def parametrize_providers(providers: list[ProviderType]):
    """Create parametrize decorator for testing multiple providers."""
    return pytest.mark.parametrize("provider_type", providers)


# Export commonly used test data
__all__ = [
    "SAMPLE_IMAGE_BASE64",
    "SAMPLE_IMAGE_BYTES",
    "LARGE_TEST_DATA_SIZE",
    "MockHelpers",
    "skip_if_no_api_key",
    "parametrize_models",
    "parametrize_providers",
]
