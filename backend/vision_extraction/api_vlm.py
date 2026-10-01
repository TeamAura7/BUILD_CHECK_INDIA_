from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from backend.vision_extraction.base import BaseArchitecturalPlanExtractor

_IMAGE_MEDIA_TYPES = {
    "jpg": "jpeg",
    "jpeg": "jpeg",
    "png": "png",
    "webp": "webp",
}


def _looks_like_unknown_param_error(message: str, param: str) -> bool:
    lowered = message.lower()
    return param.lower() in lowered and any(
        phrase in lowered for phrase in ("unknown parameter", "unsupported parameter", "unrecognized", "not supported")
    )


class ApiArchitecturalPlanExtractor(BaseArchitecturalPlanExtractor):
    """
    Hosted vision-language backend: sends each rendered page image to a
    remote, OpenAI-Chat-Completions-compatible endpoint over HTTP instead
    of loading any model weights locally.

    Exists specifically for machines that can't run `smolvlm` or `qwen`
    locally (not enough RAM/VRAM to load even the ~4.4GB SmolVLM2
    weights, or generation is too slow on CPU) -- this backend needs no
    `torch`/`transformers`/GPU at all, only stdlib `urllib`, and defers
    the actual model inference to a provider's servers.

    Works with any OpenAI-compatible vision endpoint by pointing
    `settings.vision_api_base_url` at it -- Groq (the default; this
    project already has GROQ_API_KEY configured for text/RAG), OpenAI,
    OpenRouter, Together, Fireworks, or a self-hosted OpenAI-compatible
    server (vLLM, LM Studio, Ollama's OpenAI-compatible route, etc.).

    Same prompt (`prompts.ARCHITECTURAL_PLAN_PROMPT`), same
    `VisionPageResult`/`VisionDocumentResult` schema, same downstream
    integration as `smolvlm`/`qwen` -- switching to this backend is a
    config change (`VISION_BACKEND=api`), not a code change anywhere
    else in the pipeline.
    """

    # NOTE: hosted vision-model line-ups (especially on Groq) change
    # frequently, and this may go stale -- confirm the current
    # production-ready model in the provider's docs before relying on
    # this default; override with VISION_API_MODEL if it has moved on.
    DEFAULT_MODEL_NAME = "qwen/qwen3.6-27b"

    def __init__(self, model_name: str | None = None):
        super().__init__(model_name)
        from backend.config import get_settings

        settings = get_settings()
        # A separate vision_api_model override takes precedence over the
        # generic vision_model_name (shared with the local backends),
        # since local-backend model ids (HF repo ids) and hosted API
        # model ids are different namespaces.
        self.model_name = (
            model_name or settings.vision_api_model or settings.vision_model_name or self.DEFAULT_MODEL_NAME
        )
        self._base_url = (settings.vision_api_base_url or "https://api.groq.com/openai/v1").rstrip("/")
        # Falls back to groq_api_key only when still pointed at Groq's
        # default endpoint -- pointing at a different provider without
        # setting vision_api_key should fail loudly, not silently send a
        # Groq key to someone else's server.
        if settings.vision_api_key:
            self._api_key = settings.vision_api_key
        elif self._base_url == "https://api.groq.com/openai/v1":
            self._api_key = settings.groq_api_key
        else:
            self._api_key = ""
        self._temperature = settings.vision_api_temperature
        self._json_mode = settings.vision_api_json_mode
        self._reasoning_format = settings.vision_api_reasoning_format
        self._reasoning_effort = settings.vision_api_reasoning_effort
        self._timeout = settings.vision_api_timeout_seconds
        self._max_retries = settings.vision_api_max_retries

    def _load(self) -> None:
        # Nothing to load -- this is the point. Just fail fast, before
        # rendering any pages, if the backend obviously can't be called.
        if not self._api_key:
            raise RuntimeError(
                "No API key configured for the 'api' vision backend. Set VISION_API_KEY "
                f"in your .env (base URL is currently {self._base_url!r}; if that's Groq's "
                "default endpoint, GROQ_API_KEY also works)."
            )
        if not self.model_name:
            raise RuntimeError("No model configured for the 'api' vision backend (VISION_API_MODEL).")

    @staticmethod
    def _encode_image(image_path: Path) -> tuple[str, str]:
        suffix = image_path.suffix.lower().lstrip(".")
        media_type = _IMAGE_MEDIA_TYPES.get(suffix, "png")
        data = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return media_type, data

    @staticmethod
    def _resize_for_api(image_path: Path, max_pixels: int) -> Path:
        """
        Downscale a rendered page image if it exceeds a provider's pixel
        budget, preserving aspect ratio (e.g. Groq rejects images over
        33,177,600px with HTTP 400 "Image too large" -- large-format
        architectural sheets rendered at a normal DPI blow past that
        easily). Returns `image_path` unchanged when it's already within
        budget, when `max_pixels <= 0` (disabled), or when Pillow can't
        be imported/parse the file -- never blocks a request over an
        optional size optimization.
        """
        if max_pixels <= 0:
            return image_path
        try:
            from PIL import Image
        except ImportError:
            return image_path

        try:
            with Image.open(image_path) as img:
                width, height = img.size
                pixel_count = width * height
                if pixel_count <= max_pixels:
                    return image_path
                scale = (max_pixels / pixel_count) ** 0.5
                new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
                resized = img.convert("RGB").resize(new_size, Image.LANCZOS)
                resized_path = image_path.with_name(f"{image_path.stem}_apiresized.png")
                resized.save(resized_path, format="PNG")
                return resized_path
        except Exception:
            return image_path

    def _build_payload(self, image_path: Path, prompt: str, max_new_tokens: int | None = None) -> dict:
        from backend.config import get_settings

        settings = get_settings()
        encode_path = self._resize_for_api(image_path, settings.vision_api_max_image_pixels)
        media_type, b64_data = self._encode_image(encode_path)
        payload: dict = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/{media_type};base64,{b64_data}"},
                        },
                    ],
                }
            ],
            "temperature": self._temperature,
            "max_completion_tokens": max_new_tokens or get_settings().vision_max_new_tokens,
        }
        if self._json_mode:
            payload["response_format"] = {"type": "json_object"}
        if self._reasoning_format:
            # Reasoning/"thinking" models (e.g. Groq's qwen/qwen3.6-27b)
            # otherwise burn the token budget on visible <think>...</think>
            # chain-of-thought before ever emitting the actual JSON answer
            # -- easy to mistake for a truncated/malformed response. Groq's
            # docs are explicit that 'parsed' or 'hidden' is REQUIRED when
            # combining a reasoning model with response_format json_object;
            # sending both together (rather than dropping response_format,
            # which was tried first and is wrong) is the actual fix.
            payload["reasoning_format"] = self._reasoning_format
        if self._reasoning_effort:
            # For hybrid thinking/non-thinking models (e.g. Groq's
            # qwen/qwen3.6-27b): 'none' skips the deep chain-of-thought
            # pass so the model answers directly instead of potentially
            # burning the whole vision_max_new_tokens budget on hidden
            # reasoning before ever emitting the JSON (see api_vlm._post
            # for the matching unknown-parameter fallback).
            payload["reasoning_effort"] = self._reasoning_effort
        return payload

    def _post(self, payload: dict) -> dict:
        try:
            return self._post_with_retries(payload)
        except RuntimeError as exc:
            message = str(exc)
            # Some providers don't recognize `reasoning_format` at all
            # (it's Groq-specific) and reject the whole request over an
            # unknown parameter -- drop it and retry rather than failing.
            if "reasoning_format" in payload and _looks_like_unknown_param_error(message, "reasoning_format"):
                return self._post(  # re-enter _post so the json_validate_failed branch below still applies
                    {k: v for k, v in payload.items() if k != "reasoning_format"}
                )
            # Same idea for `reasoning_effort` -- only qwen/qwen3.6-27b and a
            # couple of others support it on Groq; other models/providers
            # may reject it as an unknown parameter.
            if "reasoning_effort" in payload and _looks_like_unknown_param_error(message, "reasoning_effort"):
                return self._post(
                    {k: v for k, v in payload.items() if k != "reasoning_effort"}
                )
            # Some hosted models -- especially "preview" vision/reasoning
            # models -- don't reliably honor strict response_format={"type":
            # "json_object"} mode even with reasoning_format set, and
            # return HTTP 400 code=json_validate_failed with an empty
            # failed_generation. Retrying once in plain-text mode and
            # letting base._extract_json() pull JSON out of the raw reply
            # (already handles markdown fences / surrounding prose / a
            # leading <think>...</think> block) is more robust than
            # failing the whole page over a strict-mode quirk.
            if "json_validate_failed" in message and "response_format" in payload:
                fallback_payload = {k: v for k, v in payload.items() if k != "response_format"}
                return self._post_with_retries(fallback_payload)
            raise

    def _post_with_retries(self, payload: dict) -> dict:
        url = f"{self._base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
            # Groq's endpoint sits behind Cloudflare, which blocks the
            # default `Python-urllib/x.y` User-Agent outright (HTTP 403,
            # Cloudflare error code 1010 -- "banned based on browser
            # signature", nothing to do with the API key). A normal-
            # looking User-Agent/Accept pair avoids that, whether the
            # provider is Groq or something else Cloudflare-fronted.
            "User-Agent": "buildcheck-vision/1.0 (+https://github.com)",
            "Accept": "application/json",
        }
        body = json.dumps(payload).encode("utf-8")

        last_error: Exception | None = None
        attempts = self._max_retries + 1
        for attempt in range(1, attempts + 1):
            # Initialize retry state for every attempt. This must happen
            # before the try block because URLError/TimeoutError paths do
            # not provide a Retry-After header.
            retry_after = None
            retryable = False

            request = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self._timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"Vision API HTTP {exc.code} from {self._base_url}: {detail[:800]}")
                retryable = exc.code == 429 or exc.code >= 500
                if exc.code == 429:
                    raw_retry_after = exc.headers.get("Retry-After") if exc.headers else None
                    try:
                        retry_after = float(raw_retry_after) if raw_retry_after is not None else None
                    except (TypeError, ValueError):
                        retry_after = None
            except urllib.error.URLError as exc:
                last_error = RuntimeError(f"Vision API request to {self._base_url} failed: {exc.reason}")
                retryable = True
            except TimeoutError as exc:
                last_error = RuntimeError(f"Vision API request to {self._base_url} timed out: {exc}")
                retryable = True

            if not retryable or attempt >= attempts:
                raise last_error
            # For a 429 with no explicit Retry-After header, the previous
            # min(2**attempt, 20) backoff (2s, 4s, ...) was tuned for a
            # generic transient failure, not a genuinely per-MINUTE token
            # quota -- confirmed the real cause of most 429s seen from this
            # project's hosted-API testing (e.g. Groq's output-tokens-per-
            # minute limit). A page's region-focused Vision passes (site
            # plan, area statement, elevation/section height) make several
            # sequential calls sharing that one budget, so the LAST call in
            # the sequence (usually height) is the one most likely to hit
            # an already-exhausted quota -- a multi-second backoff never
            # gives it a real chance to refill; scaling toward the actual
            # ~60s window does.
            delay = retry_after if retry_after is not None else min(15.0 * attempt, 60.0)
            time.sleep(min(max(delay, 0.5), 60.0))

        raise last_error or RuntimeError("Vision API request failed for an unknown reason.")

    def _generate_raw_response(self, image_path: Path, prompt: str, max_new_tokens: int | None = None) -> str:
        payload = self._build_payload(image_path, prompt, max_new_tokens=max_new_tokens)
        response = self._post(payload)
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"Unexpected response shape from vision API ({self._base_url}): {response!r}"
            ) from exc