"""Optional deep-CV hooks for architectural drawing geometry.

A generic torchvision detector is deliberately NOT used as the final source of
truth: it has no architectural-plan classes unless domain weights are supplied.
This module provides a safe hook for a project-specific PyTorch line/segmentation
model while keeping deterministic OpenCV as the default fallback.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def run_domain_model(image: Any, weights_path: str | Path | None = None) -> dict:
    """Run an optional project-specific TorchScript/model hook.

    The weights must be trained for architectural-plan segmentation/line detection.
    If no weights are configured, return an explicit non-result so callers continue
    with OpenCV LSD/contours instead of hallucinating generic object detections.
    """
    if not weights_path:
        return {"enabled": False, "reason": "No domain-specific PyTorch weights configured."}
    try:
        import torch
        model = torch.jit.load(str(weights_path), map_location="cpu")
        model.eval()
        with torch.no_grad():
            output = model(image)
        return {"enabled": True, "output": output}
    except Exception as exc:
        return {"enabled": False, "reason": f"Domain PyTorch model unavailable: {exc}"}
