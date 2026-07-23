import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import structlog

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from transformers import AutoTokenizer, PreTrainedTokenizerFast

logger = structlog.get_logger(__name__)


class TokenizerManager:
    def __init__(self, model_name: str, seed: int | None = None) -> None:
        self._model_name = model_name
        self._rng = np.random.default_rng(seed)
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
        except ValueError as exc:
            if "Tokenizer class TokenizersBackend" not in str(exc):
                raise
            self._tokenizer = self._load_tokenizers_backend_fallback(model_name)

    @staticmethod
    def _load_tokenizers_backend_fallback(
        model_name: str,
    ) -> PreTrainedTokenizerFast:
        """Load transformers-v5 TokenizersBackend artifacts on transformers v4.

        GLM-5.2 declares ``TokenizersBackend`` in tokenizer_config.json. That
        generic class exists in transformers v5, while trie intentionally pins
        v4. Loading the same tokenizer.json and chat template directly avoids a
        model-specific tokenizer dependency and keeps replay tokenization equal
        to the server.
        """
        model_path = Path(model_name)
        tokenizer_path = model_path / "tokenizer.json"
        config_path = model_path / "tokenizer_config.json"
        template_path = model_path / "chat_template.jinja"
        if not tokenizer_path.is_file() or not config_path.is_file():
            raise ValueError(
                "TokenizersBackend fallback requires a local model directory "
                "containing tokenizer.json and tokenizer_config.json"
            )

        with config_path.open("r", encoding="utf-8") as config_file:
            config = json.load(config_file)
        kwargs: dict[str, Any] = {
            "tokenizer_file": str(tokenizer_path),
            "clean_up_tokenization_spaces": config.get(
                "clean_up_tokenization_spaces", False
            ),
        }
        for field in ("bos_token", "eos_token", "unk_token", "pad_token"):
            value = config.get(field)
            if isinstance(value, (str, dict)):
                kwargs[field] = value
        model_max_length = config.get("model_max_length")
        if isinstance(model_max_length, int):
            kwargs["model_max_length"] = model_max_length
        if template_path.is_file():
            kwargs["chat_template"] = template_path.read_text(encoding="utf-8")

        logger.info(
            "loading generic TokenizersBackend artifact with "
            "PreTrainedTokenizerFast",
            model=model_name,
        )
        return PreTrainedTokenizerFast(**kwargs)

    def _decode(self, token_ids: list[int]) -> str:
        return self._tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    def _encode(self, text: str) -> list[int]:
        return self._tokenizer(text, add_special_tokens=False).input_ids

    def encode(self, text: str) -> list[int]:
        return self._encode(text)

    def decode(self, token_ids: list[int]) -> str:
        return self._decode(token_ids)

    def count_tokens(self, text: str) -> int:
        return len(self._encode(text))

    def render_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        add_generation_prompt: bool = True,
    ) -> str:
        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": add_generation_prompt,
        }
        if tools is not None:
            kwargs["tools"] = tools
        rendered = self._tokenizer.apply_chat_template(messages, **kwargs)
        if not isinstance(rendered, str):
            raise TypeError("chat template did not return text")
        return rendered

    def _sample_token_ids(self, prompt_length: int) -> list[int]:
        return self._rng.integers(
            self._tokenizer.vocab_size, size=prompt_length
        ).tolist()

    def get_prompt(
        self,
        prompt_length: int,
        *,
        max_rounds: int = 10,
    ) -> str:
        """Return text that round-trips to ``prompt_length`` tokens, repairing
        decode/encode mismatches for up to ``max_rounds`` iterations."""
        if prompt_length == 0:
            return ""

        token_ids = self._sample_token_ids(prompt_length)
        prompt = self._decode(token_ids)
        retokenized_ids = self._encode(prompt)

        rounds = 0
        while len(retokenized_ids) != prompt_length and rounds < max_rounds:
            if len(retokenized_ids) < prompt_length:
                num_extras = prompt_length - len(retokenized_ids)
                token_ids = retokenized_ids + self._sample_token_ids(num_extras)
            else:
                token_ids = retokenized_ids[:prompt_length]
            prompt = self._decode(token_ids)
            retokenized_ids = self._encode(prompt)
            rounds += 1

        if len(retokenized_ids) != prompt_length:
            logger.warning(
                "Failed to synthesize prompt for "
                f"{self._model_name!r}: target_length={prompt_length}, "
                f"final_length={len(retokenized_ids)}, repair_rounds={rounds}, "
                f"max_rounds={max_rounds}. Returning best effort result."
            )
            return prompt

        return prompt
