from __future__ import annotations

from pathlib import Path

from backend.vision_extraction.base import BaseArchitecturalPlanExtractor


class QwenArchitecturalPlanExtractor(BaseArchitecturalPlanExtractor):
    """
    Qwen2.5-VL-7B-Instruct semantic extractor. Heavier than the default
    `smolvlm` backend (~16GB of weights vs ~4.4GB) -- prefer `smolvlm`
    unless you have a specific reason to want Qwen's larger model
    (more GPU/CPU RAM, more download time and disk space, more
    latency per page).
    """

    DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

    def __init__(self, model_name: str | None = None):
        super().__init__(model_name)
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return

        try:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "Qwen vision dependencies are not installed. "
                "Install them with: pip install -r requirements-vision-qwen.txt"
            ) from exc

        print(f"Loading Qwen2.5-VL model weights ({self.model_name}, this can take a while)...", flush=True)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto",
        )
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        print(f"Model loaded on device: {self._model.device}", flush=True)

    def _generate_raw_response(self, image_path: Path, prompt: str, max_new_tokens: int | None = None) -> str:
        from qwen_vl_utils import process_vision_info

        from backend.config import get_settings

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        )
        inputs = inputs.to(self._model.device)

        max_new_tokens = max_new_tokens or get_settings().vision_max_new_tokens
        generated_ids = self._model.generate(**inputs, max_new_tokens=max_new_tokens)
        trimmed = [
            generated_ids[i][len(inputs.input_ids[i]):]
            for i in range(len(generated_ids))
        ]
        return self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
