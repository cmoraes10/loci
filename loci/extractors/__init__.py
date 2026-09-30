"""Optional model-backed extractors.

These live outside core/ because they make HTTP calls and carry provider
configuration. Inject the result as a ModelExtractor into extract().
"""

from .model import ModelExtractorConfig, build_extractor

__all__ = ["ModelExtractorConfig", "build_extractor"]
