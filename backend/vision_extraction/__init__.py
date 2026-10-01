from __future__ import annotations

from backend.config import get_settings
from backend.schemas.vision import (
    VisionArea,
    VisionDimension,
    VisionDocumentResult,
    VisionPageResult,
    VisionRegion,
)
from backend.vision_extraction.api_vlm import ApiArchitecturalPlanExtractor
from backend.vision_extraction.base import BaseArchitecturalPlanExtractor
from backend.vision_extraction.qwen_vl import QwenArchitecturalPlanExtractor
from backend.vision_extraction.smolvlm import SmolVLMArchitecturalPlanExtractor

_BACKENDS: dict[str, type[BaseArchitecturalPlanExtractor]] = {
    "smolvlm": SmolVLMArchitecturalPlanExtractor,
    "qwen": QwenArchitecturalPlanExtractor,
    "api": ApiArchitecturalPlanExtractor,
}


def get_vision_extractor() -> BaseArchitecturalPlanExtractor:
    """
    Return the configured vision extractor backend
    (`settings.vision_backend`, default `"smolvlm"`). This is the single
    place that decides which VLM implementation is used --
    `pdf_extractor.py` and `run_vision.py` both go through this instead
    of importing a specific backend class directly, so switching models
    (e.g. to `qwen`, or to the hosted `api` backend for machines that
    can't run a local VLM) is a config change, not a code change.
    """
    settings = get_settings()
    backend = (settings.vision_backend or "smolvlm").lower()
    try:
        extractor_cls = _BACKENDS[backend]
    except KeyError as exc:
        raise ValueError(
            f"Unknown vision_backend {backend!r}. Valid options: {sorted(_BACKENDS)}"
        ) from exc
    return extractor_cls()


__all__ = [
    "ApiArchitecturalPlanExtractor",
    "BaseArchitecturalPlanExtractor",
    "QwenArchitecturalPlanExtractor",
    "SmolVLMArchitecturalPlanExtractor",
    "get_vision_extractor",
    "VisionArea",
    "VisionDimension",
    "VisionDocumentResult",
    "VisionPageResult",
    "VisionRegion",
]
