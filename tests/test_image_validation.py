"""Tests for provider-independent image validation."""

import base64
import os
from unittest.mock import Mock, patch

import pytest

from tests.mock_helpers import MinimalTestProvider
from tests.test_helpers import ImageValidationAssertions, create_test_data_urls, create_test_error_patterns


class TestImageValidation:
    """Test suite for image validation functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        # Create a minimal concrete provider instance for testing base class methods
        self.provider = MinimalTestProvider(api_key="test-key")

    def test_validate_data_url_valid(self, test_image_data, test_image_data_url) -> None:
        """Test validation of valid data URL."""
        ImageValidationAssertions.assert_image_validation_success(
            self.provider, test_image_data_url, test_image_data, "image/png"
        )

    @pytest.mark.parametrize(
        "url_key,error_key",
        [
            ("invalid_format", "invalid_format"),  # Missing base64 part
            ("missing_data", "invalid_format"),  # Missing data
            ("unsupported_type", "unsupported_type"),  # Not an image
        ],
    )
    def test_validate_data_url_invalid_format(self, url_key: str, error_key: str) -> None:
        """Test validation of malformed data URL."""
        data_urls = create_test_data_urls()
        error_patterns = create_test_error_patterns()

        ImageValidationAssertions.assert_data_url_error(
            self.provider, data_urls[url_key], error_patterns[error_key]
        )

    def test_non_data_url_treated_as_file_path(self) -> None:
        """Test that non-data URLs are treated as file paths."""
        error_patterns = create_test_error_patterns()
        ImageValidationAssertions.assert_file_validation_error(
            self.provider, "image/png;base64,abc123", error_patterns["file_not_found"]
        )

    def test_validate_data_url_unsupported_type(self) -> None:
        """Test validation of unsupported image type in data URL."""
        data_urls = create_test_data_urls()
        error_patterns = create_test_error_patterns()

        ImageValidationAssertions.assert_data_url_error(
            self.provider, data_urls["unsupported_bmp"], error_patterns["unsupported_bmp"]
        )

    def test_validate_data_url_invalid_base64(self) -> None:
        """Test validation of data URL with invalid base64."""
        data_urls = create_test_data_urls()
        error_patterns = create_test_error_patterns()

        ImageValidationAssertions.assert_data_url_error(
            self.provider, data_urls["invalid_base64"], error_patterns["invalid_base64"]
        )

    def test_validate_large_data_url(self, large_test_data) -> None:
        """Test validation of large data URL to ensure size limits work."""
        # Encode as base64 and create data URL
        encoded_data = base64.b64encode(large_test_data).decode()
        data_url = f"data:image/png;base64,{encoded_data}"

        # Should fail with default 20MB limit
        with pytest.raises(ValueError) as excinfo:
            self.provider.validate_image(data_url)
        assert "Image too large: 21.0MB (max: 20.0MB)" in str(excinfo.value)

        # Should succeed with higher limit
        ImageValidationAssertions.assert_image_validation_success(
            self.provider, data_url, large_test_data, "image/png", max_size_mb=25.0
        )

    def test_validate_file_path_valid(self, temp_image_file, test_image_data) -> None:
        """Test validation of valid image file."""
        ImageValidationAssertions.assert_image_validation_success(
            self.provider, temp_image_file, test_image_data, "image/png"
        )

    def test_validate_file_path_not_found(self) -> None:
        """Test validation of non-existent file."""
        error_patterns = create_test_error_patterns()
        ImageValidationAssertions.assert_file_validation_error(
            self.provider, "/path/to/nonexistent/image.png", error_patterns["file_not_found"]
        )

    def test_validate_file_path_unsupported_extension(self, temp_file_factory) -> None:
        """Test validation of file with unsupported extension."""
        bmp_file = temp_file_factory(data=b"dummy data", suffix=".bmp")
        ImageValidationAssertions.assert_file_validation_error(
            self.provider, bmp_file, "Unsupported image format: .bmp"
        )

    def test_validate_file_path_read_error(self, temp_file_factory) -> None:
        """Test validation when file cannot be read."""
        import os

        png_file = temp_file_factory(suffix=".png")
        # Remove the file but keep the path
        os.unlink(png_file)

        error_patterns = create_test_error_patterns()
        ImageValidationAssertions.assert_file_validation_error(
            self.provider, png_file, error_patterns["file_not_found"]
        )

    def test_validate_image_size_limit(self, temp_file_factory, large_test_data) -> None:
        """Test validation of image size limits."""
        large_file = temp_file_factory(data=large_test_data, suffix=".png")

        with pytest.raises(ValueError) as excinfo:
            self.provider.validate_image(large_file, max_size_mb=20.0)
        assert "Image too large: 21.0MB (max: 20.0MB)" in str(excinfo.value)

    def test_validate_image_custom_size_limit(self, temp_file_factory) -> None:
        """Test validation with custom size limit."""
        # Create a 2MB "image"
        data = b"x" * (2 * 1024 * 1024)
        png_file = temp_file_factory(data=data, suffix=".png")

        # Should fail with 1MB limit
        with pytest.raises(ValueError) as excinfo:
            self.provider.validate_image(png_file, max_size_mb=1.0)
        assert "Image too large: 2.0MB (max: 1.0MB)" in str(excinfo.value)

        # Should succeed with 3MB limit
        ImageValidationAssertions.assert_image_validation_success(
            self.provider, png_file, data, "image/png", max_size_mb=3.0
        )

    def test_validate_image_default_size_limit(self, temp_file_factory) -> None:
        """Test validation with default size limit (None)."""
        # Create a small image that's under the default limit
        data = b"x" * (1024 * 1024)  # 1MB
        jpg_file = temp_file_factory(data=data, suffix=".jpg")

        # Should succeed with default limit (20MB)
        ImageValidationAssertions.assert_image_validation_success(
            self.provider, jpg_file, data, "image/jpeg"
        )

        # Should also succeed when explicitly passing None
        ImageValidationAssertions.assert_image_validation_success(
            self.provider, jpg_file, data, "image/jpeg", max_size_mb=None
        )

    @pytest.mark.parametrize(
        "ext,expected_mime",
        [
            (".png", "image/png"),
            (".jpg", "image/jpeg"),
            (".jpeg", "image/jpeg"),
            (".gif", "image/gif"),
            (".webp", "image/webp"),
        ],
    )
    def test_validate_all_supported_formats(self, temp_file_factory, ext, expected_mime) -> None:
        """Test validation of all supported image formats."""
        data = b"dummy image data"
        image_file = temp_file_factory(data=data, suffix=ext)

        ImageValidationAssertions.assert_image_validation_success(
            self.provider, image_file, data, expected_mime
        )


class TestProviderIntegration:
    """Test image validation integration with different providers."""

    @patch("utils.gemini_errors.logger")
    def test_gemini_provider_uses_validation(self, mock_logger: Mock) -> None:
        """Test that Gemini provider uses the base validation."""
        from providers.gemini import GeminiModelProvider

        # Create a provider instance
        provider = GeminiModelProvider(api_key="test-key")

        # Test with non-existent file
        result = provider._process_image("/nonexistent/image.png")
        assert result is None
        mock_logger.warning.assert_called_with("Image file not found: /nonexistent/image.png")

    @patch("providers.openai_compatible.logging")
    def test_openai_compatible_provider_uses_validation(self, mock_logging: Mock) -> None:
        """Test that OpenAI-compatible providers use the base validation."""
        from providers.xai import XAIModelProvider

        # Create a provider instance (XAI inherits from OpenAICompatibleProvider)
        provider = XAIModelProvider(api_key="test-key")

        # Test with non-existent file
        result = provider._process_image("/nonexistent/image.png")
        assert result is None
        mock_logging.warning.assert_called_with("Image file not found: /nonexistent/image.png")

    def test_data_url_preservation(self) -> None:
        """Test that data URLs are properly preserved through validation."""
        from providers.xai import XAIModelProvider

        provider = XAIModelProvider(api_key="test-key")

        # Valid data URL
        data_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="

        result = provider._process_image(data_url)
        assert result is not None
        assert result["type"] == "image_url"
        assert result["image_url"]["url"] == data_url

    @patch.dict(os.environ, {"VERTEX_AI_PROJECT_ID": "test-project"})
    @patch("utils.credential_manager.default")
    def test_vertex_ai_provider_uses_validation(self, mock_default: Mock) -> None:
        """Test that Vertex AI provider uses the base validation."""
        # Mock credentials
        mock_credentials = Mock()
        mock_default.return_value = (mock_credentials, None)

        from providers.vertex_ai import VertexAIModelProvider

        # Create a provider instance
        provider = VertexAIModelProvider(api_key="test-key")

        # Test with non-existent file
        with pytest.raises(ValueError, match="Image file not found"):
            provider.validate_image("/nonexistent/image.png")
