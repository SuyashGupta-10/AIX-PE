"""Qwen3 / Qwen3-VL access layer.

Two backends:
  * "openai"       - any OpenAI-compatible endpoint serving Qwen3 (Ollama `qwen3:4b`,
                     vLLM, LM Studio, OpenRouter, Alibaba DashScope ...)
  * "transformers" - load Qwen3 locally from the Hugging Face hub.

Qwen3 is a hybrid "thinking" model. For a triage bot we want fast, direct
answers, so thinking is switched off (`enable_thinking=False` locally, `/no_think`
+ chat_template_kwargs on servers) and any stray <think> block is stripped.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import re
import threading
import time

from . import config

log = logging.getLogger("llm")

_THINK = re.compile(r"<think>.*?</think>", re.S)


def _clean(text: str) -> str:
    text = _THINK.sub("", text or "")
    return text.replace("<think>", "").replace("</think>", "").strip()


def parse_json(text: str):
    """Pull the first JSON object out of a model reply."""
    if not text:
        return None
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


class _OpenAIBackend:
    def __init__(self):
        from openai import OpenAI

        self.client = OpenAI(base_url=config.QWEN_BASE_URL, api_key=config.QWEN_API_KEY, timeout=config.LLM_TIMEOUT)

    def chat(self, messages, max_tokens, temperature):
        msgs = [dict(m) for m in messages]
        if msgs and msgs[-1]["role"] == "user" and isinstance(msgs[-1]["content"], str):
            msgs[-1]["content"] += " /no_think"
        r = self.client.chat.completions.create(
            model=config.QWEN_MODEL,
            messages=msgs,
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return r.choices[0].message.content

    def vision(self, image_bytes, mime, prompt, max_tokens):
        url = f"data:{mime};base64," + base64.b64encode(image_bytes).decode()
        r = self.client.chat.completions.create(
            model=config.QWEN_VL_MODEL,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": url}},
                {"type": "text", "text": prompt},
            ]}],
            max_tokens=max_tokens,
            temperature=0.1,
        )
        return r.choices[0].message.content


class _TransformersBackend:
    def __init__(self):
        import torch

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # fp32 is usually faster than bf16 on consumer CPUs
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self._text = None
        self._vl = None
        self._lock = threading.Lock()

    def _load_text(self):
        if self._text is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            t0 = time.time()
            log.info("Loading %s on %s ...", config.QWEN_MODEL, self.device)
            tok = AutoTokenizer.from_pretrained(config.QWEN_MODEL)
            model = AutoModelForCausalLM.from_pretrained(config.QWEN_MODEL, dtype=self.dtype).to(self.device).eval()
            self._text = (tok, model)
            log.info("Loaded in %.1fs", time.time() - t0)
        return self._text

    def _load_vl(self):
        if self._vl is None:
            from transformers import AutoModelForImageTextToText, AutoProcessor

            log.info("Loading %s on %s ...", config.QWEN_VL_MODEL, self.device)
            proc = AutoProcessor.from_pretrained(config.QWEN_VL_MODEL)
            model = AutoModelForImageTextToText.from_pretrained(config.QWEN_VL_MODEL, dtype=self.dtype).to(self.device).eval()
            self._vl = (proc, model)
        return self._vl

    def chat(self, messages, max_tokens, temperature):
        with self._lock:
            tok, model = self._load_text()
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            inputs = tok(prompt, return_tensors="pt").to(self.device)
            gen = dict(max_new_tokens=max_tokens, do_sample=temperature > 0)
            if temperature > 0:
                gen.update(temperature=temperature, top_p=0.8, top_k=20)
            with self.torch.inference_mode():
                out = model.generate(**inputs, **gen)
            return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def vision(self, image_bytes, mime, prompt, max_tokens):
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img.thumbnail((768, 768))
        with self._lock:
            proc, model = self._load_vl()
            messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
            inputs = proc.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                              return_dict=True, return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
            return proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]


class QwenClient:
    def __init__(self):
        self.backend_name = config.QWEN_BACKEND
        self._backend = None
        self.last_error: str | None = None
        if self.backend_name == "openai":
            self._backend = _OpenAIBackend()
        elif self.backend_name == "transformers":
            self._backend = _TransformersBackend()

    @property
    def enabled(self) -> bool:
        return self._backend is not None

    @property
    def on_gpu(self) -> bool:
        return isinstance(self._backend, _TransformersBackend) and self._backend.device == "cuda"

    @property
    def vision_enabled(self) -> bool:
        return self.enabled and bool(config.QWEN_VL_MODEL)

    def chat(self, messages, max_tokens=None, temperature=0.2) -> str | None:
        if not self.enabled:
            return None
        try:
            t0 = time.time()
            out = _clean(self._backend.chat(messages, max_tokens or config.MAX_NEW_TOKENS, temperature))
            log.info("qwen chat %.1fs", time.time() - t0)
            self.last_error = None
            return out
        except Exception as e:  # network down, model missing, OOM ...
            self.last_error = f"{type(e).__name__}: {e}"
            log.warning("Qwen call failed, falling back to rules: %s", self.last_error)
            return None

    def vision(self, image_bytes: bytes, mime: str, prompt: str, max_tokens=400) -> str | None:
        if not self.vision_enabled:
            return None
        try:
            return _clean(self._backend.vision(image_bytes, mime, prompt, max_tokens))
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            log.warning("Qwen-VL call failed: %s", self.last_error)
            return None


_client: QwenClient | None = None


def get_client() -> QwenClient:
    global _client
    if _client is None:
        _client = QwenClient()
    return _client
