"""Shared error handling patterns for Gemini-based providers.

This module provides shared error handling utilities that were previously
duplicated between the Direct Gemini API provider and Vertex AI provider.
It extracts the ~70 lines of duplicated error handling logic into reusable utilities.
"""

import logging
from typing import Union

logger = logging.getLogger(__name__)


def is_gemini_error_retryable(error: Exception) -> bool:
    """Determine if a Gemini API error should be retried.

    Uses Gemini API error structure instead of text pattern matching for reliability.
    This function consolidates the retry logic that was duplicated between providers.

    Args:
        error: Exception from Gemini API call

    Returns:
        True if error should be retried, False otherwise
    """
    error_str = str(error).lower()

    # Check for 429 errors first - these need special handling
    if "429" in error_str or "quota" in error_str or "resource_exhausted" in error_str:
        # For Gemini, check for specific non-retryable error indicators
        # These typically indicate permanent failures or quota/size limits
        non_retryable_indicators = [
            "quota exceeded",
            "resource exhausted",
            "context length",
            "token limit",
            "request too large",
            "invalid request",
            "quota_exceeded",
            "resource_exhausted",
        ]

        # Also check if this is a structured error from Gemini SDK
        try:
            # Try to access error details if available
            if hasattr(error, "details") or hasattr(error, "reason"):
                # Gemini API errors may have structured details
                error_details = getattr(error, "details", "") or getattr(error, "reason", "")
                error_details_str = str(error_details).lower()

                # Check for non-retryable error codes/reasons
                if any(indicator in error_details_str for indicator in non_retryable_indicators):
                    logger.debug(f"Non-retryable Gemini error: {error_details}")
                    return False
        except Exception:
            pass

        # Check main error string for non-retryable patterns
        if any(indicator in error_str for indicator in non_retryable_indicators):
            logger.debug(f"Non-retryable Gemini error based on message: {error_str[:200]}...")
            return False

        # If it's a 429/quota error but doesn't match non-retryable patterns, it might be retryable rate limiting
        logger.debug(f"Retryable Gemini rate limiting error: {error_str[:100]}...")
        return True

    # For non-429 errors, check if they're retryable
    retryable_indicators = [
        "timeout",
        "connection",
        "network",
        "temporary",
        "unavailable",
        "retry",
        "internal error",
        "408",  # Request timeout
        "500",  # Internal server error
        "502",  # Bad gateway
        "503",  # Service unavailable
        "504",  # Gateway timeout
        "ssl",  # SSL errors
        "handshake",  # Handshake failures
    ]

    return any(indicator in error_str for indicator in retryable_indicators)


def process_gemini_image(image_path: str, validate_image_func) -> Union[dict, None]:
    """Process an image for Gemini API using provider's validation function.

    This consolidates the image processing logic that was duplicated between providers.

    Args:
        image_path: Path to image file or data URL
        validate_image_func: Provider's image validation function

    Returns:
        Image data dict for Gemini API, or None if processing failed
    """
    try:
        # Use provider's validation function
        image_bytes, mime_type = validate_image_func(image_path)

        # For data URLs, extract the base64 data directly
        if image_path.startswith("data:"):
            # Extract base64 data from data URL
            _, data = image_path.split(",", 1)
            return {"inline_data": {"mime_type": mime_type, "data": data}}
        else:
            # For file paths, encode the bytes
            import base64

            image_data = base64.b64encode(image_bytes).decode()
            return {"inline_data": {"mime_type": mime_type, "data": image_data}}

    except ValueError as e:
        logger.warning(str(e))
        return None
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")
        return None


def check_vision_support(model_name: str) -> bool:
    """Check if a Gemini model supports vision (image processing).

    This uses the shared registry as source of truth to avoid duplication.

    Args:
        model_name: The model name to check

    Returns:
        True if model supports vision, False otherwise
    """
    # Use the registry as source of truth
    from providers.gemini_core import GeminiModelRegistry

    spec = GeminiModelRegistry.get_spec(model_name)
    return spec.supports_images if spec else False


def extract_gemini_usage(response) -> dict[str, int]:
    """Extract token usage from Gemini response.

    This consolidates the usage extraction logic that was similar between providers.

    Args:
        response: Gemini API response object

    Returns:
        Dictionary with usage information
    """
    usage = {}

    # Try to extract usage metadata from response
    # Note: The actual structure depends on the SDK version and response format
    if hasattr(response, "usage_metadata"):
        metadata = response.usage_metadata

        # Extract token counts with explicit None checks
        input_tokens = None
        output_tokens = None

        if hasattr(metadata, "prompt_token_count"):
            value = metadata.prompt_token_count
            if value is not None:
                input_tokens = value
                usage["input_tokens"] = value

        if hasattr(metadata, "candidates_token_count"):
            value = metadata.candidates_token_count
            if value is not None:
                output_tokens = value
                usage["output_tokens"] = value

        # Calculate total only if both values are available and valid
        if input_tokens is not None and output_tokens is not None:
            usage["total_tokens"] = input_tokens + output_tokens

    return usage


def estimate_tokens_fallback(text: str) -> int:
    """Fallback token estimation for Gemini models.

    This provides consistent token estimation across providers.

    Args:
        text: Text to estimate tokens for

    Returns:
        Estimated token count
    """
    # Simple estimation: ~4 characters per token for English text
    return len(text) // 4
