"""
Logging conventions for BUILDCheck India.

Rules every teammate should follow:

1. Never use `print()`. Use `get_logger(__name__)`.
2. Log messages describing a compliance-relevant decision must include
   `plan_id` and, where applicable, `rule_id` in `extra=` so logs are
   traceable back to a specific plan/rule without grepping free text.
3. Extraction/normalization warnings that affect confidence should be
   logged at WARNING and ALSO appended to the relevant model's
   `warnings` / `conflict` field — logs are for debugging, the model
   fields are the audit trail.
4. Never log API keys, full file contents, or PII beyond what's already
   in the plan being processed.
"""

from __future__ import annotations

import logging
import sys

from backend.config import get_settings

_CONFIGURED = False


def _configure_root() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    settings = get_settings()
    handler = logging.StreamHandler(stream=sys.stdout)
    if settings.log_json:
        fmt = (
            '{"time":"%(asctime)s","level":"%(levelname)s",'
            '"logger":"%(name)s","message":"%(message)s"}'
        )
    else:
        fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    handler.setFormatter(logging.Formatter(fmt))

    root = logging.getLogger("buildcheck")
    root.setLevel(settings.log_level.upper())
    root.addHandler(handler)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger, e.g. get_logger(__name__)."""
    _configure_root()
    return logging.getLogger(f"buildcheck.{name}")
