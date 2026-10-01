"""
Tests for the vision_extraction backend factory/base class added when
swapping the default vision backend from Qwen2.5-VL-7B-Instruct to the
much lighter SmolVLM2-2.2B-Instruct (see PHASE3_1_NOTES.md).

These deliberately do NOT load real model weights (no GPU/large
download in CI) -- they cover the parts that are actually regression-
prone: backend selection, model-name resolution precedence, and the
raw-model-text -> JSON parsing shared by every backend.
"""

from __future__ import annotations

import base64
import io as _io
import json

import pytest

from backend.config import get_settings
from backend.vision_extraction import (
    ApiArchitecturalPlanExtractor,
    QwenArchitecturalPlanExtractor,
    SmolVLMArchitecturalPlanExtractor,
    get_vision_extractor,
)
from backend.vision_extraction.base import BaseArchitecturalPlanExtractor


def test_backends_are_concrete_subclasses_of_the_shared_base():
    assert issubclass(SmolVLMArchitecturalPlanExtractor, BaseArchitecturalPlanExtractor)
    assert issubclass(QwenArchitecturalPlanExtractor, BaseArchitecturalPlanExtractor)
    assert issubclass(ApiArchitecturalPlanExtractor, BaseArchitecturalPlanExtractor)
    # Constructing the wrapper must not require model weights -- _load() is lazy.
    SmolVLMArchitecturalPlanExtractor()
    QwenArchitecturalPlanExtractor()
    ApiArchitecturalPlanExtractor()


def test_default_backend_is_smolvlm(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "vision_backend", "smolvlm")
    assert isinstance(get_vision_extractor(), SmolVLMArchitecturalPlanExtractor)


def test_get_vision_extractor_can_select_qwen_backend(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "vision_backend", "qwen")
    assert isinstance(get_vision_extractor(), QwenArchitecturalPlanExtractor)


def test_get_vision_extractor_can_select_api_backend(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "vision_backend", "api")
    assert isinstance(get_vision_extractor(), ApiArchitecturalPlanExtractor)


def test_get_vision_extractor_rejects_unknown_backend(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "vision_backend", "not-a-real-backend")
    with pytest.raises(ValueError):
        get_vision_extractor()


def test_backend_default_model_name_used_when_settings_unset(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "vision_model_name", "")
    extractor = SmolVLMArchitecturalPlanExtractor()
    assert extractor.model_name == "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

    extractor = QwenArchitecturalPlanExtractor()
    assert extractor.model_name == "Qwen/Qwen2.5-VL-7B-Instruct"


def test_explicit_constructor_model_name_wins_over_everything():
    extractor = SmolVLMArchitecturalPlanExtractor(model_name="custom/explicit-override")
    assert extractor.model_name == "custom/explicit-override"


def test_settings_model_name_overrides_backend_default(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "vision_model_name", "custom/from-settings")
    extractor = QwenArchitecturalPlanExtractor()
    assert extractor.model_name == "custom/from-settings"


def test_extract_json_handles_markdown_code_fences():
    text = '```json\n{"regions": [], "dimensions": []}\n```'
    assert BaseArchitecturalPlanExtractor._extract_json(text) == {"regions": [], "dimensions": []}


def test_extract_json_handles_surrounding_prose():
    text = 'Sure, here is the result: {"a": 1, "b": [1, 2]} -- hope that helps!'
    assert BaseArchitecturalPlanExtractor._extract_json(text) == {"a": 1, "b": [1, 2]}


def test_extract_json_raises_on_genuinely_unparseable_text():
    with pytest.raises(Exception):
        BaseArchitecturalPlanExtractor._extract_json("no json anywhere in this response")


def test_vision_max_new_tokens_setting_is_generous_enough_to_avoid_silent_truncation():
    settings = get_settings()
    # A too-low budget doesn't just truncate output, it breaks extraction
    # outright: a busy real-world multi-view sheet's regions+dimensions+
    # areas JSON can run well past 2000 tokens, and a response cut off
    # before its closing brace fails BaseArchitecturalPlanExtractor.
    # _extract_json entirely, dropping that page's vision evidence. The
    # default needs enough headroom for that, while still being
    # overridable downward via VISION_MAX_NEW_TOKENS for a quick CPU
    # smoke test on a simple page.
    assert 2500 <= settings.vision_max_new_tokens <= 6000


# --- 'api' backend: hosted VLM over HTTP, no local weights ---------------


def test_api_backend_load_fails_fast_without_a_key(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "vision_api_key", "")
    monkeypatch.setattr(settings, "groq_api_key", "")
    extractor = ApiArchitecturalPlanExtractor()
    with pytest.raises(RuntimeError, match="No API key"):
        extractor._load()


def test_api_backend_falls_back_to_groq_key_on_groq_default_endpoint(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "vision_api_key", "")
    monkeypatch.setattr(settings, "vision_api_base_url", "https://api.groq.com/openai/v1")
    monkeypatch.setattr(settings, "groq_api_key", "groq-secret")
    extractor = ApiArchitecturalPlanExtractor()
    assert extractor._api_key == "groq-secret"
    extractor._load()  # must not raise


def test_api_backend_does_not_leak_groq_key_to_other_providers(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "vision_api_key", "")
    monkeypatch.setattr(settings, "vision_api_base_url", "https://api.openai.com/v1")
    monkeypatch.setattr(settings, "groq_api_key", "groq-secret")
    extractor = ApiArchitecturalPlanExtractor()
    assert extractor._api_key == ""
    with pytest.raises(RuntimeError, match="No API key"):
        extractor._load()


def test_api_backend_model_name_precedence(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "vision_api_model", "")
    monkeypatch.setattr(settings, "vision_model_name", "")
    assert ApiArchitecturalPlanExtractor().model_name == ApiArchitecturalPlanExtractor.DEFAULT_MODEL_NAME

    monkeypatch.setattr(settings, "vision_model_name", "shared/override")
    assert ApiArchitecturalPlanExtractor().model_name == "shared/override"

    monkeypatch.setattr(settings, "vision_api_model", "api-specific/override")
    assert ApiArchitecturalPlanExtractor().model_name == "api-specific/override"

    assert ApiArchitecturalPlanExtractor(model_name="explicit").model_name == "explicit"


def test_api_backend_encode_image_picks_media_type_from_suffix(tmp_path):
    png_path = tmp_path / "page.png"
    png_path.write_bytes(b"fake-png-bytes")
    media_type, data = ApiArchitecturalPlanExtractor._encode_image(png_path)
    assert media_type == "png"
    assert data  # base64 payload, non-empty

    jpg_path = tmp_path / "page.jpg"
    jpg_path.write_bytes(b"fake-jpg-bytes")
    media_type, _ = ApiArchitecturalPlanExtractor._encode_image(jpg_path)
    assert media_type == "jpeg"


def test_api_backend_resize_for_api_leaves_small_images_untouched(tmp_path):
    from PIL import Image

    image_path = tmp_path / "page.png"
    Image.new("RGB", (100, 80), color="white").save(image_path)

    result = ApiArchitecturalPlanExtractor._resize_for_api(image_path, max_pixels=33_177_600)
    assert result == image_path


def test_api_backend_resize_for_api_downscales_oversized_images(tmp_path):
    from PIL import Image

    image_path = tmp_path / "page.png"
    # 9000x5600 = 50,400,000px, mirroring the real-world Groq 400 error
    # ("images can contain at most 33177600 pixels but image contained
    # 50400000") from a large-format sheet rendered at 200 DPI.
    Image.new("RGB", (9000, 5600), color="white").save(image_path)

    result = ApiArchitecturalPlanExtractor._resize_for_api(image_path, max_pixels=33_177_600)
    assert result != image_path
    assert result.exists()
    with Image.open(result) as resized:
        assert resized.width * resized.height <= 33_177_600
        # aspect ratio preserved (within float rounding)
        assert abs((resized.width / resized.height) - (9000 / 5600)) < 0.01


def test_api_backend_resize_for_api_disabled_when_max_pixels_is_zero(tmp_path):
    from PIL import Image

    image_path = tmp_path / "page.png"
    Image.new("RGB", (9000, 5600), color="white").save(image_path)

    result = ApiArchitecturalPlanExtractor._resize_for_api(image_path, max_pixels=0)
    assert result == image_path


def test_api_backend_resize_for_api_tolerates_unparseable_files(tmp_path):
    bogus_path = tmp_path / "not_really_an_image.png"
    bogus_path.write_bytes(b"not a real png")
    # Must not raise -- falls back to the original path rather than
    # blocking the request over an optional size optimization.
    result = ApiArchitecturalPlanExtractor._resize_for_api(bogus_path, max_pixels=100)
    assert result == bogus_path


def test_api_backend_build_payload_downscales_oversized_images(monkeypatch, tmp_path):
    from PIL import Image

    settings = get_settings()
    monkeypatch.setattr(settings, "vision_api_max_image_pixels", 1000)
    image_path = tmp_path / "page.png"
    Image.new("RGB", (200, 200), color="white").save(image_path)  # 40,000px > 1000px budget

    extractor = ApiArchitecturalPlanExtractor(model_name="test-model")
    payload = extractor._build_payload(image_path, "prompt")
    content = payload["messages"][0]["content"]
    url = content[1]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")

    b64_data = url.split(",", 1)[1]
    decoded = base64.b64decode(b64_data)
    with Image.open(_io.BytesIO(decoded)) as resized:
        assert resized.width * resized.height <= 1000
    settings = get_settings()
    monkeypatch.setattr(settings, "vision_api_json_mode", True)
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"fake-png-bytes")

    extractor = ApiArchitecturalPlanExtractor(model_name="test-model")
    payload = extractor._build_payload(image_path, "describe this plan")

    assert payload["model"] == "test-model"
    assert payload["response_format"] == {"type": "json_object"}
    content = payload["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "describe this plan"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_api_backend_generate_raw_response_parses_openai_shaped_reply(monkeypatch, tmp_path):
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"fake-png-bytes")
    extractor = ApiArchitecturalPlanExtractor(model_name="test-model")

    def fake_post(self, payload):
        assert payload["model"] == "test-model"
        return {"choices": [{"message": {"content": '{"regions": []}'}}]}

    monkeypatch.setattr(ApiArchitecturalPlanExtractor, "_post", fake_post)
    result = extractor._generate_raw_response(image_path, "prompt")
    assert result == '{"regions": []}'


def test_api_backend_generate_raw_response_raises_on_unexpected_shape(monkeypatch, tmp_path):
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"fake-png-bytes")
    extractor = ApiArchitecturalPlanExtractor(model_name="test-model")

    monkeypatch.setattr(ApiArchitecturalPlanExtractor, "_post", lambda self, payload: {"unexpected": True})
    with pytest.raises(RuntimeError, match="Unexpected response shape"):
        extractor._generate_raw_response(image_path, "prompt")


def test_api_backend_post_retries_on_5xx_then_succeeds(monkeypatch, tmp_path):
    import urllib.error

    settings = get_settings()
    monkeypatch.setattr(settings, "vision_api_max_retries", 2)
    extractor = ApiArchitecturalPlanExtractor(model_name="test-model")
    extractor._api_key = "k"

    calls = {"n": 0}

    class _FakeResponse:
        def __init__(self, body: bytes):
            self._body = body

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(request, timeout=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.HTTPError(
                url="https://example.test", code=503, msg="busy", hdrs=None, fp=_io.BytesIO(b"server busy")
            )
        return _FakeResponse(json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode("utf-8"))

    monkeypatch.setattr("time.sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    result = extractor._post({"model": "test-model", "messages": []})
    assert result["choices"][0]["message"]["content"] == "ok"
    assert calls["n"] == 3


def test_api_backend_post_gives_up_after_max_retries(monkeypatch):
    import urllib.error

    settings = get_settings()
    monkeypatch.setattr(settings, "vision_api_max_retries", 1)
    extractor = ApiArchitecturalPlanExtractor(model_name="test-model")
    extractor._api_key = "k"

    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(
            url="https://example.test", code=429, msg="rate limited", hdrs=None, fp=_io.BytesIO(b"slow down")
        )

    monkeypatch.setattr("time.sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="Vision API HTTP 429"):
        extractor._post({"model": "test-model", "messages": []})


def test_api_backend_post_does_not_retry_on_4xx_client_errors(monkeypatch):
    import urllib.error

    settings = get_settings()
    monkeypatch.setattr(settings, "vision_api_max_retries", 3)
    extractor = ApiArchitecturalPlanExtractor(model_name="test-model")
    extractor._api_key = "k"

    calls = {"n": 0}

    def fake_urlopen(request, timeout=None):
        calls["n"] += 1
        raise urllib.error.HTTPError(
            url="https://example.test", code=401, msg="unauthorized", hdrs=None, fp=_io.BytesIO(b"bad key")
        )

    monkeypatch.setattr("time.sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="Vision API HTTP 401"):
        extractor._post({"model": "test-model", "messages": []})
    assert calls["n"] == 1  # no retry burned on a non-retryable client error


def test_api_backend_post_sends_a_browser_like_user_agent(monkeypatch):
    # Regression test: Groq's endpoint sits behind Cloudflare, which
    # returns HTTP 403 (Cloudflare error code 1010) for the default
    # `Python-urllib/x.y` User-Agent, independent of whether the API key
    # is valid. Every outgoing request must set a non-default User-Agent.
    extractor = ApiArchitecturalPlanExtractor(model_name="test-model")
    extractor._api_key = "k"

    captured = {}

    class _FakeResponse:
        def read(self):
            return json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(request, timeout=None):
        captured["headers"] = dict(request.header_items())
        return _FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    extractor._post({"model": "test-model", "messages": []})

    user_agent = next((v for k, v in captured["headers"].items() if k.lower() == "user-agent"), None)
    assert user_agent is not None
    assert "python-urllib" not in user_agent.lower()
