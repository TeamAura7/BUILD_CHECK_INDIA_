from __future__ import annotations

from pathlib import Path

from backend.vision_extraction.base import BaseArchitecturalPlanExtractor


class SmolVLMArchitecturalPlanExtractor(BaseArchitecturalPlanExtractor):
    """
    SmolVLM2-2.2B-Instruct semantic extractor (default vision backend).

    Chosen over Qwen2.5-VL-7B-Instruct for a much smaller download
    (~4.4GB vs ~16GB safetensors -- less exposure to interrupted/flaky
    downloads) and lower memory footprint (~5GB GPU RAM vs 16GB+),
    while still being instruction-tuned well enough to follow this
    project's structured-JSON extraction prompt. It is a genuinely
    weaker model than Qwen2.5-VL-7B on raw benchmarks, so if semantic
    region/dimension accuracy is insufficient in practice, the `qwen`
    backend remains available (`VISION_BACKEND=qwen`).

    Native `transformers` support (`AutoModelForImageTextToText`), no
    extra `*_vl_utils` package, no `trust_remote_code`.
    """

    DEFAULT_MODEL_NAME = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

    def __init__(self, model_name: str | None = None):
        super().__init__(model_name)
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Vision dependencies are not installed. Install them with: "
                "pip install -r requirements-vision.txt"
            ) from exc

        print(f"Loading SmolVLM2 processor ({self.model_name})...", flush=True)
        self._processor = AutoProcessor.from_pretrained(self.model_name)

        # flash_attention_2 gives a real speedup but requires a matching
        # CUDA build of flash-attn, which isn't always installable
        # (e.g. CPU-only machines, some Windows setups). Try it, and
        # fall back to the default attention implementation rather than
        # failing extraction entirely over a missing optional speedup.
        print("Loading SmolVLM2 model weights (this can take a while on CPU)...", flush=True)
        try:
            self._model = AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                _attn_implementation="flash_attention_2",
            )
        except Exception:
            logger_msg = (
                "flash_attention_2 unavailable for SmolVLM2 (this is normal on "
                "CPU-only or non-CUDA setups); falling back to default attention."
            )
            from backend.tools.logging_config import get_logger

            get_logger(__name__).info(logger_msg)
            print(logger_msg, flush=True)
            self._model = AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        print(f"Model loaded on device: {self._model.device}", flush=True)

    def _generate_raw_response(self, image_path: Path, prompt: str, max_new_tokens: int | None = None) -> str:
        import torch

        from backend.config import get_settings

        # A plain absolute local path string is accepted directly by
        # transformers' image loader (`transformers.image_utils.load_image`)
        # used under the hood by the processor's chat template -- no need
        # for a `file://` prefix or a pre-loaded PIL.Image.
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "url": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }]

        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[1]
        max_new_tokens = max_new_tokens or get_settings().vision_max_new_tokens
        generated_ids = self._model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
        trimmed = generated_ids[:, input_len:]
        return self._processor.batch_decode(trimmed, skip_special_tokens=True)[0]
