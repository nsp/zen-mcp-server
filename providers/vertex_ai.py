"""Vertex AI provider."""

import base64
import io
import logging
import mimetypes
import os
import threading
from typing import Any, Optional, Union

from google.api_core import exceptions, retry
from google.cloud import aiplatform
from vertexai.generative_models import Content, GenerationConfig, GenerativeModel, Part

from .base import (
    ModelCapabilities,
    ModelProvider,
    ModelResponse,
    ProviderType,
    create_temperature_constraint,
)
from .gemini_core import GeminiModelRegistry

logger = logging.getLogger(__name__)


class VertexAIModelProvider(ModelProvider):
    """Vertex AI provider."""

    def __init__(self, api_key: str = "", **kwargs):
        """Initialize with lazy initialization pattern."""
        super().__init__(api_key, **kwargs)

        # Lazy initialization state
        self._initialized = False
        self._init_lock = threading.Lock()

        # Will be set during lazy initialization
        self.credentials = None
        self.project_id = None
        self.location = None
        self.claude_location = None
        self.gemini_models = None
        self.models = None

    # Claude models (coding-suitable models from available list)
    CLAUDE_MODELS = {
        "claude-opus-4": ModelCapabilities(
            provider=ProviderType.VERTEX_AI,
            model_name="claude-opus-4",
            friendly_name="Claude Opus 4 (Vertex AI)",
            context_window=200_000,
            max_output_tokens=8_192,
            supports_system_prompts=True,
            supports_images=True,
            supports_temperature=True,
            temperature_constraint=create_temperature_constraint("range"),
            aliases=["vertex-claude-opus-4", "vertex-opus-4"],
        ),
        "claude-sonnet-4": ModelCapabilities(
            provider=ProviderType.VERTEX_AI,
            model_name="claude-sonnet-4",
            friendly_name="Claude Sonnet 4 (Vertex AI)",
            context_window=200_000,
            max_output_tokens=8_192,
            supports_system_prompts=True,
            supports_images=True,
            supports_temperature=True,
            temperature_constraint=create_temperature_constraint("range"),
            aliases=["vertex-claude-sonnet-4", "vertex-sonnet-4"],
        ),
    }

    def _lazy_init(self):
        """Initialize provider resources on first use."""
        if self._initialized:
            return

        with self._init_lock:
            # Double-check pattern
            if self._initialized:
                return

            try:
                # Import required modules
                import google.auth
                from google.cloud import aiplatform

                # Auto-detect credentials and project
                self.credentials, self.project_id = google.auth.default()

                # Get configuration from environment
                self.location = os.getenv("VERTEX_AI_LOCATION", "us-central1")
                self.claude_location = os.getenv("VERTEX_AI_CLAUDE_LOCATION", "us-east5")

                # Override project if specified in env
                if env_project := os.getenv("VERTEX_AI_PROJECT_ID"):
                    self.project_id = env_project

                if not self.project_id:
                    raise ValueError("VERTEX_AI_PROJECT_ID required or must be auto-detected from credentials")

                # Initialize Vertex AI SDK
                aiplatform.init(project=self.project_id, location=self.location, credentials=self.credentials)

                # Create models using shared registry
                self.gemini_models = GeminiModelRegistry.create_provider_models(
                    provider_type=ProviderType.VERTEX_AI,
                    provider_overrides={
                        "gemini-2.5-pro": {
                            "aliases": ["vertex-gemini-pro", "vertex-pro"],
                            "friendly_name_override": "Gemini 2.5 Pro (Vertex AI)",
                        },
                        "gemini-2.5-flash": {
                            "aliases": ["vertex-gemini-flash", "vertex-flash"],
                            "friendly_name_override": "Gemini 2.5 Flash (Vertex AI)",
                        },
                    },
                )

                # Combine all models
                self.models = {**self.gemini_models, **self.CLAUDE_MODELS}

                # Add aliases (create a copy to avoid modifying during iteration)
                alias_mapping = {}
                for _model_name, capabilities in self.models.items():
                    for alias in capabilities.aliases:
                        alias_mapping[alias] = capabilities

                # Update models with aliases
                self.models.update(alias_mapping)

                logger.info(f"Initialized Vertex AI provider with {len(self.models)} models")
                self._initialized = True

            except Exception as e:
                logger.error(f"Failed to initialize Vertex AI provider: {e}")
                raise

    def get_provider_type(self) -> ProviderType:
        """Get provider type."""
        return ProviderType.VERTEX_AI

    def list_models(self, respect_restrictions: bool = True) -> dict:
        """List available models."""
        self._lazy_init()
        return self.models.copy()

    def validate_model_name(self, model_name: str) -> bool:
        """Validate model name."""
        self._lazy_init()
        return model_name in self.models

    def get_capabilities(self, model_name: str) -> ModelCapabilities:
        """Get model capabilities."""
        self._lazy_init()
        if model_name not in self.models:
            raise ValueError(f"Unsupported model: {model_name}")
        return self.models[model_name]

    def supports_thinking_mode(self, model_name: str) -> bool:
        """Check if model supports thinking mode."""
        # Claude models don't support thinking mode
        if self._is_claude_model(model_name):
            return False

        # Use shared utility for Gemini models
        from utils.thinking_mode import validate_thinking_mode_support

        return validate_thinking_mode_support(self._resolve_model_name(model_name), "medium")

    def count_tokens(self, text: str, model_name: str) -> int:
        """Count tokens with estimation fallback."""
        self._lazy_init()
        # For Gemini models, try to use API; for Claude, estimate
        if self._is_claude_model(model_name):
            return len(text) // 4

        try:
            from vertexai.generative_models import GenerativeModel

            model = GenerativeModel(self._resolve_model_name(model_name))
            return model.count_tokens(text).total_tokens
        except Exception:
            return len(text) // 4

    def generate_content(
        self,
        prompt: str,
        model_name: str = "gemini-2.5-flash",
        temperature: float = 0.7,
        max_output_tokens: Optional[int] = None,
        thinking_mode: Optional[str] = None,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
        images: Optional[list[Union[str, bytes]]] = None,
        **kwargs,
    ) -> ModelResponse:
        """Generate content with automatic retries."""
        self._lazy_init()
        if self._is_claude_model(model_name):
            return self._generate_claude(
                prompt, model_name, temperature, max_output_tokens, system_prompt, json_mode, images, **kwargs
            )
        else:
            return self._generate_gemini(
                prompt,
                model_name,
                temperature,
                max_output_tokens,
                thinking_mode,
                system_prompt,
                json_mode,
                images,
                **kwargs,
            )

    def _generate_gemini(
        self,
        prompt: str,
        model_name: str,
        temperature: float,
        max_output_tokens: Optional[int],
        thinking_mode: Optional[str],
        system_prompt: Optional[str],
        json_mode: bool,
        images: Optional[list[Union[str, bytes]]],
        **kwargs,
    ) -> ModelResponse:
        """Generate with Gemini models."""

        resolved_model = self._resolve_model_name(model_name)

        # Create model with system instruction
        model = GenerativeModel(
            model_name=resolved_model,
            system_instruction=system_prompt,
        )

        # Prepare content
        content_parts = []

        # Add images if provided
        if images:
            for image in images:
                if image_part := self._process_image(image):
                    content_parts.append(image_part)

        # Add text
        content_parts.append(Part.from_text(prompt))

        # Generate with config
        config = GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_mime_type="application/json" if json_mode else "text/plain",
        )

        # Apply retry decorator dynamically
        retry_decorator = retry.Retry(
            predicate=retry.if_exception_type(
                (
                    exceptions.ServiceUnavailable,
                    exceptions.DeadlineExceeded,
                    exceptions.ResourceExhausted,
                )
            ),
            initial=1.0,
            maximum=32.0,
            multiplier=2.0,
        )

        @retry_decorator
        def _generate_with_retry():
            return model.generate_content(
                Content(role="user", parts=content_parts),
                generation_config=config,
            )

        response = _generate_with_retry()

        # Extract usage
        usage = {}
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = {
                "input_tokens": getattr(response.usage_metadata, "prompt_token_count", 0),
                "output_tokens": getattr(response.usage_metadata, "candidates_token_count", 0),
            }
            if usage["input_tokens"] or usage["output_tokens"]:
                usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]

        return ModelResponse(
            content=response.text,
            usage=usage,
            model_name=resolved_model,
            provider=self.get_provider_type(),
        )

    def _generate_claude(
        self,
        prompt: str,
        model_name: str,
        temperature: float,
        max_output_tokens: Optional[int],
        system_prompt: Optional[str],
        json_mode: bool,
        images: Optional[list[Union[str, bytes]]],
        **kwargs,
    ) -> ModelResponse:
        """Generate with Claude models via Vertex AI."""

        resolved_model = self._resolve_model_name(model_name)

        # Use aiplatform Model for Claude
        model_path = (
            f"projects/{self.project_id}/locations/{self.claude_location}/publishers/anthropic/models/{resolved_model}"
        )
        model = aiplatform.Model(model_path)

        # Build message content
        content = []
        if images:
            for image in images:
                if image_data := self._process_claude_image(image):
                    content.append(image_data)
        content.append({"type": "text", "text": prompt})

        # Build request
        instance = {
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_output_tokens or 8192,
            "temperature": temperature,
        }

        if system_prompt:
            instance["system"] = system_prompt

        # Make prediction
        response = model.predict(instances=[instance])
        prediction = response.predictions[0]

        # Extract content
        content_text = ""
        if "content" in prediction:
            for item in prediction["content"]:
                if item.get("type") == "text":
                    content_text = item.get("text", "")
                    break

        # Extract usage
        usage = {}
        if "usage" in prediction:
            usage_data = prediction["usage"]
            usage = {
                "input_tokens": usage_data.get("input_tokens", 0),
                "output_tokens": usage_data.get("output_tokens", 0),
            }
            usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]

        return ModelResponse(
            content=content_text,
            usage=usage,
            model_name=resolved_model,
            provider=self.get_provider_type(),
        )

    def _process_image(self, image: Union[str, bytes]) -> Optional:
        """Process image for Gemini using PIL."""
        try:
            from PIL import Image
            from vertexai.generative_models import Part

            if isinstance(image, str):
                if image.startswith("data:"):
                    # Data URL - extract base64 data
                    header, data = image.split(",", 1)
                    mime_type = header.split(";")[0].split(":")[1]
                    image_bytes = base64.b64decode(data)
                else:
                    # File path
                    mime_type, _ = mimetypes.guess_type(image)
                    if not mime_type or not mime_type.startswith("image/"):
                        logger.warning(f"Unsupported image type: {mime_type}")
                        return None

                    with open(image, "rb") as f:
                        image_bytes = f.read()
            else:
                # Raw bytes
                image_bytes = image
                mime_type = "image/png"  # Default assumption

            # Validate with PIL
            with Image.open(io.BytesIO(image_bytes)) as img:
                # Convert to supported format if needed
                if img.format.lower() not in ["png", "jpeg", "webp", "gif"]:
                    output = io.BytesIO()
                    img.save(output, format="PNG")
                    image_bytes = output.getvalue()
                    mime_type = "image/png"

            return Part.from_data(data=image_bytes, mime_type=mime_type)

        except Exception as e:
            logger.warning(f"Failed to process image: {e}")
            return None

    def _process_claude_image(self, image: Union[str, bytes]) -> Optional[dict[str, Any]]:
        """Process image for Claude format."""
        try:
            if isinstance(image, str) and image.startswith("data:"):
                # Data URL
                header, data = image.split(",", 1)
                mime_type = header.split(";")[0].split(":")[1]
                return {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime_type,
                        "data": data,
                    },
                }
            elif isinstance(image, str):
                # File path
                with open(image, "rb") as f:
                    image_bytes = f.read()
            else:
                # Raw bytes
                image_bytes = image

            # Encode as base64
            encoded = base64.b64encode(image_bytes).decode()
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": encoded,
                },
            }

        except Exception as e:
            logger.warning(f"Failed to process Claude image: {e}")
            return None

    def _is_claude_model(self, model_name: str) -> bool:
        """Check if model is Claude."""
        self._lazy_init()
        resolved = self._resolve_model_name(model_name)
        return resolved in self.CLAUDE_MODELS

    def _resolve_model_name(self, model_name: str) -> str:
        """Resolve alias to canonical name."""
        self._lazy_init()
        # If it's already canonical, return it
        if model_name in self.gemini_models or model_name in self.CLAUDE_MODELS:
            return model_name

        # Check if it's an alias
        if model_name in self.models:
            capabilities = self.models[model_name]
            return capabilities.model_name

        return model_name

    def close(self):
        """Cleanup resources."""
        pass
