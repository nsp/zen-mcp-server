"""Shared test utilities for common testing patterns."""

import pytest


class ImageValidationAssertions:
    """Assertion helpers for image validation tests."""

    @staticmethod
    def assert_data_url_error(provider, data_url, expected_error):
        """Assert that a data URL raises the expected validation error."""
        with pytest.raises(ValueError) as excinfo:
            provider.validate_image(data_url)
        assert expected_error in str(excinfo.value)

    @staticmethod
    def assert_file_validation_error(provider, file_path, expected_error):
        """Assert that a file path raises the expected validation error."""
        with pytest.raises(ValueError) as excinfo:
            provider.validate_image(file_path)
        assert expected_error in str(excinfo.value)

    @staticmethod
    def assert_image_validation_success(provider, image_input, expected_data, expected_mime, **kwargs):
        """Assert that image validation succeeds with expected results."""
        image_bytes, mime_type = provider.validate_image(image_input, **kwargs)
        assert image_bytes == expected_data
        assert mime_type == expected_mime


def create_test_data_urls():
    """Create standard test data URLs for validation testing."""
    return {
        "valid_png": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
        "invalid_format": "data:image/png",  # Missing base64 part
        "missing_data": "data:image/png;base64",  # Missing data
        "unsupported_type": "data:text/plain;base64,dGVzdA==",  # Not an image
        "unsupported_bmp": "data:image/bmp;base64,Qk0=",  # BMP format
        "invalid_base64": "data:image/png;base64,@@@invalid@@@",
    }


def create_test_error_patterns():
    """Create standard error patterns for testing."""
    return {
        "invalid_format": "Invalid data URL format",
        "unsupported_type": "Unsupported image type",
        "unsupported_bmp": "Unsupported image type: image/bmp",
        "invalid_base64": "Invalid base64 data",
        "file_not_found": "Image file not found",
        "unsupported_extension": "Unsupported image format",
        "too_large": "Image too large",
    }
