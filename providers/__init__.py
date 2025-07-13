"""Model provider abstractions for supporting multiple AI providers."""

from .base import ModelCapabilities, ModelProvider, ModelResponse
from .gemini import GeminiModelProvider
from .openai_compatible import OpenAICompatibleProvider
from .openai_provider import OpenAIModelProvider
from .openrouter import OpenRouterProvider
from .registry import ModelProviderRegistry

# Optional import for VertexAI (requires additional dependencies)
try:
    from .vertex_ai import VertexAIModelProvider

    _vertex_ai_available = True
except ImportError:
    VertexAIModelProvider = None
    _vertex_ai_available = False

__all__ = [
    "ModelProvider",
    "ModelResponse",
    "ModelCapabilities",
    "ModelProviderRegistry",
    "GeminiModelProvider",
    "OpenAIModelProvider",
    "OpenAICompatibleProvider",
    "OpenRouterProvider",
]

# Conditionally add VertexAI if available
if _vertex_ai_available:
    __all__.append("VertexAIModelProvider")
