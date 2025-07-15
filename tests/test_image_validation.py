"""Tests for provider-independent image validation using optimized factory patterns."""

import base64
import os
import tempfile
from typing import Optional, Dict, Any
from unittest.mock import Mock, patch

import pytest

from providers.base import ModelCapabilities, ModelProvider, ModelResponse, ProviderType
from conftest import SAMPLE_IMAGE_BASE64, SAMPLE_IMAGE_BYTES, LARGE_TEST_DATA_SIZE


class MinimalTestProvider(ModelProvider):
    """Minimal concrete provider for testing base class methods."""

    def get_capabilities(self, model_name: str) -> ModelCapabilities:
        """Not needed for image validation tests."""
        raise NotImplementedError("Not needed for image validation tests")

    def generate_content(
        self,
        prompt: str,
        model_name: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_output_tokens: Optional[int] = None,
        **kwargs,
    ) -> ModelResponse:
        """Not needed for image validation tests."""
        raise NotImplementedError("Not needed for image validation tests")

    def count_tokens(self, text: str, model_name: str) -> int:
        """Not needed for image validation tests."""
        raise NotImplementedError("Not needed for image validation tests")

    def get_provider_type(self) -> ProviderType:
        """Not needed for image validation tests."""
        raise NotImplementedError("Not needed for image validation tests")

    def validate_model_name(self, model_name: str) -> bool:
        """Not needed for image validation tests."""
        raise NotImplementedError("Not needed for image validation tests")

    def supports_thinking_mode(self, model_name: str) -> bool:
        """Not needed for image validation tests."""
        raise NotImplementedError("Not needed for image validation tests")


# Test data matrices for parameterized testing
DATA_URL_ERROR_MATRIX = [
    ("invalid_format", "invalid_format"),  # Missing base64 part
    ("missing_data", "invalid_format"),    # Missing data
    ("unsupported_type", "unsupported_type"),  # Not an image
    ("unsupported_bmp", "unsupported_bmp"),   # BMP format
    ("invalid_base64", "invalid_base64"),     # Invalid base64
]

FILE_PATH_ERROR_MATRIX = [
    ("/path/to/nonexistent/image.png", "file_not_found"),
    (".bmp", "unsupported_extension"),  # Will be created with temp_file_factory
]

SUPPORTED_FORMATS_MATRIX = [
    (".png", "image/png"),
    (".jpg", "image/jpeg"),
    (".jpeg", "image/jpeg"),
    (".gif", "image/gif"),
    (".webp", "image/webp"),
]

SIZE_LIMIT_MATRIX = [
    (1, 2, False),  # 2MB file with 1MB limit should fail
    (3, 2, True),   # 2MB file with 3MB limit should succeed
    (None, 1, True),  # 1MB file with default limit should succeed
]

PROVIDER_INTEGRATION_MATRIX = [
    ("providers.gemini", "GeminiModelProvider", "utils.gemini_errors.logger"),
    ("providers.xai", "XAIModelProvider", "providers.openai_compatible.logging"),
]


def create_test_data_urls() -> Dict[str, str]:
    """Create test data URLs for various error conditions."""
    return {
        "invalid_format": "data:image/png",  # Missing base64 part
        "missing_data": "data:image/png;base64",  # Missing data
        "unsupported_type": "data:text/plain;base64,dGVzdA==",  # Not an image
        "unsupported_bmp": "data:image/bmp;base64,Qk0=",  # BMP format
        "invalid_base64": "data:image/png;base64,@@@invalid@@@",  # Invalid base64
    }


def create_test_error_patterns() -> Dict[str, str]:
    """Create expected error patterns for validation."""
    return {
        "invalid_format": "Invalid data URL format",
        "unsupported_type": "Unsupported image type",
        "unsupported_bmp": "Unsupported image type: image/bmp",
        "invalid_base64": "Invalid base64 data",
        "file_not_found": "Image file not found",
        "unsupported_extension": "Unsupported image format",
    }


@pytest.fixture
def image_provider():
    """Fixture for creating test image provider instances."""
    return MinimalTestProvider(api_key="test-key")


@pytest.fixture
def test_image_data():
    """Fixture for sample image data."""
    return SAMPLE_IMAGE_BYTES


@pytest.fixture
def test_image_data_url(test_image_data):
    """Fixture for sample image data URL."""
    return f"data:image/png;base64,{base64.b64encode(test_image_data).decode()}"


@pytest.fixture
def large_test_data():
    """Fixture for large test data (21MB)."""
    return b"x" * LARGE_TEST_DATA_SIZE


@pytest.fixture
def temp_image_file(test_image_data):
    """Fixture for temporary image file."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        tmp_file.write(test_image_data)
        tmp_file_path = tmp_file.name
    
    yield tmp_file_path
    
    # Cleanup
    try:
        os.unlink(tmp_file_path)
    except FileNotFoundError:
        pass  # File may have been deleted by test


@pytest.fixture
def temp_file_factory():
    """Factory fixture for creating temporary files with various configurations."""
    created_files = []
    
    def _create_file(data: bytes = b"dummy data", suffix: str = ".png") -> str:
        """Create a temporary file with specified data and suffix."""
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
            tmp_file.write(data)
            tmp_file_path = tmp_file.name
        
        created_files.append(tmp_file_path)
        return tmp_file_path
    
    yield _create_file
    
    # Cleanup all created files
    for file_path in created_files:
        try:
            os.unlink(file_path)
        except FileNotFoundError:
            pass  # File may have been deleted by test


@pytest.fixture
def validation_test_data():
    """Fixture for all validation test data."""
    return {
        "urls": create_test_data_urls(),
        "errors": create_test_error_patterns(),
    }


class ImageValidationAssertions:
    """Helper class for common image validation assertions."""
    
    @staticmethod
    def assert_image_validation_success(
        provider: ModelProvider, 
        image_input: str, 
        expected_data: bytes, 
        expected_mime: str,
        max_size_mb: Optional[float] = None
    ) -> None:
        """Assert successful image validation."""
        image_bytes, mime_type = provider.validate_image(image_input, max_size_mb=max_size_mb)
        assert image_bytes == expected_data
        assert mime_type == expected_mime
    
    @staticmethod
    def assert_data_url_error(provider: ModelProvider, data_url: str, expected_error: str) -> None:
        """Assert data URL validation error."""
        with pytest.raises(ValueError) as excinfo:
            provider.validate_image(data_url)
        assert expected_error in str(excinfo.value)
    
    @staticmethod
    def assert_file_validation_error(provider: ModelProvider, file_path: str, expected_error: str) -> None:
        """Assert file validation error."""
        with pytest.raises(ValueError) as excinfo:
            provider.validate_image(file_path)
        assert expected_error in str(excinfo.value)


class TestImageValidation:
    """Test suite for image validation functionality."""

    def test_validate_data_url_valid(self, image_provider, test_image_data, test_image_data_url):
        """Test validation of valid data URL."""
        ImageValidationAssertions.assert_image_validation_success(
            image_provider, test_image_data_url, test_image_data, "image/png"
        )

    @pytest.mark.parametrize("url_key,error_key", DATA_URL_ERROR_MATRIX)
    def test_validate_data_url_errors(self, image_provider, validation_test_data, url_key, error_key):
        """Test validation of malformed and unsupported data URLs."""
        ImageValidationAssertions.assert_data_url_error(
            image_provider, 
            validation_test_data["urls"][url_key], 
            validation_test_data["errors"][error_key]
        )

    def test_non_data_url_treated_as_file_path(self, image_provider, validation_test_data):
        """Test that non-data URLs are treated as file paths."""
        ImageValidationAssertions.assert_file_validation_error(
            image_provider, "image/png;base64,abc123", validation_test_data["errors"]["file_not_found"]
        )

    def test_validate_large_data_url(self, image_provider, large_test_data):
        """Test validation of large data URL to ensure size limits work."""
        # Encode as base64 and create data URL
        encoded_data = base64.b64encode(large_test_data).decode()
        data_url = f"data:image/png;base64,{encoded_data}"

        # Should fail with default 20MB limit
        with pytest.raises(ValueError) as excinfo:
            image_provider.validate_image(data_url)
        assert "Image too large: 21.0MB (max: 20.0MB)" in str(excinfo.value)

        # Should succeed with higher limit
        ImageValidationAssertions.assert_image_validation_success(
            image_provider, data_url, large_test_data, "image/png", max_size_mb=25.0
        )

    def test_validate_file_path_valid(self, image_provider, temp_image_file, test_image_data):
        """Test validation of valid image file."""
        ImageValidationAssertions.assert_image_validation_success(
            image_provider, temp_image_file, test_image_data, "image/png"
        )

    def test_validate_file_path_not_found(self, image_provider, validation_test_data):
        """Test validation of non-existent file."""
        ImageValidationAssertions.assert_file_validation_error(
            image_provider, "/path/to/nonexistent/image.png", validation_test_data["errors"]["file_not_found"]
        )

    def test_validate_file_path_unsupported_extension(self, image_provider, temp_file_factory):
        """Test validation of file with unsupported extension."""
        bmp_file = temp_file_factory(data=b"dummy data", suffix=".bmp")
        ImageValidationAssertions.assert_file_validation_error(
            image_provider, bmp_file, "Unsupported image format: .bmp"
        )

    def test_validate_file_path_read_error(self, image_provider, temp_file_factory, validation_test_data):
        """Test validation when file cannot be read."""
        png_file = temp_file_factory(suffix=".png")
        # Remove the file but keep the path
        os.unlink(png_file)

        ImageValidationAssertions.assert_file_validation_error(
            image_provider, png_file, validation_test_data["errors"]["file_not_found"]
        )

    @pytest.mark.parametrize("max_size_mb,file_size_mb,should_succeed", SIZE_LIMIT_MATRIX)
    def test_validate_image_size_limits(self, image_provider, temp_file_factory, max_size_mb, file_size_mb, should_succeed):
        """Test validation with various size limits."""
        # Create file with specified size
        data = b"x" * (file_size_mb * 1024 * 1024)
        test_file = temp_file_factory(data=data, suffix=".png")

        if should_succeed:
            ImageValidationAssertions.assert_image_validation_success(
                image_provider, test_file, data, "image/png", max_size_mb=max_size_mb
            )
        else:
            with pytest.raises(ValueError) as excinfo:
                image_provider.validate_image(test_file, max_size_mb=max_size_mb)
            assert f"Image too large: {file_size_mb}.0MB (max: {max_size_mb}.0MB)" in str(excinfo.value)

    def test_validate_image_size_limit_edge_case(self, image_provider, temp_file_factory, large_test_data):
        """Test validation of large data URL to ensure size limits work."""
        large_file = temp_file_factory(data=large_test_data, suffix=".png")
        
        with pytest.raises(ValueError) as excinfo:
            image_provider.validate_image(large_file, max_size_mb=20.0)
        assert "Image too large: 21.0MB (max: 20.0MB)" in str(excinfo.value)

    @pytest.mark.parametrize("ext,expected_mime", SUPPORTED_FORMATS_MATRIX)
    def test_validate_all_supported_formats(self, image_provider, temp_file_factory, ext, expected_mime):
        """Test validation of all supported image formats."""
        data = b"dummy image data"
        image_file = temp_file_factory(data=data, suffix=ext)
        
        ImageValidationAssertions.assert_image_validation_success(
            image_provider, image_file, data, expected_mime
        )


class TestProviderIntegration:
    """Test image validation integration with different providers."""

    @pytest.mark.parametrize("module_name,class_name,logger_path", PROVIDER_INTEGRATION_MATRIX)
    def test_provider_uses_validation(self, module_name, class_name, logger_path):
        """Test that providers use the base validation."""
        import importlib
        
        with patch(logger_path) as mock_logger:
            # Dynamically import the provider
            module = importlib.import_module(module_name)
            provider_class = getattr(module, class_name)
            provider = provider_class(api_key="test-key")

            # Test with non-existent file
            result = provider._process_image("/nonexistent/image.png")
            assert result is None
            mock_logger.warning.assert_called_with("Image file not found: /nonexistent/image.png")

    def test_data_url_preservation(self):
        """Test that data URLs are properly preserved through validation."""
        from providers.xai import XAIModelProvider

        provider = XAIModelProvider(api_key="test-key")
        data_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="

        result = provider._process_image(data_url)
        assert result is not None
        assert result["type"] == "image_url"
        assert result["image_url"]["url"] == data_url

    @patch.dict(os.environ, {"VERTEX_AI_PROJECT_ID": "test-project"})
    @patch("google.auth.default")
    @patch("google.cloud.aiplatform.init")
    def test_vertex_ai_provider_uses_validation(self, mock_init, mock_auth):
        """Test that Vertex AI provider uses the base validation."""
        mock_auth.return_value = (Mock(), "test-project")
        
        from providers.vertex_ai import VertexAIModelProvider
        provider = VertexAIModelProvider(api_key="test-key")

        # Test with non-existent file
        with pytest.raises(ValueError, match="Image file not found"):
            provider.validate_image("/nonexistent/image.png")