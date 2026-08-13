import copy
import json
import os
from typing import Any

import numpy as np
import structlog

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from transformers import AutoTokenizer

logger = structlog.get_logger(__name__)


class TokenizerManager:
    def __init__(self, model_name: str, seed: int | None = None) -> None:
        self._model_name = model_name
        self._rng = np.random.default_rng(seed)
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

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

    @staticmethod
    def _normalize_tool_call_arguments(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """OpenAI 录制格式的 tool_calls.arguments 是 JSON 字符串，
        MiniMax 模板对其调 .items()，要求 dict——这里归一化。"""
        normalized: list[dict[str, Any]] = []
        for message in messages:
            tool_calls = message.get("tool_calls")
            if not tool_calls:
                normalized.append(message)
                continue
            message = copy.deepcopy(message)
            for tool_call in message.get("tool_calls") or []:
                function = tool_call.get("function") or tool_call
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    try:
                        function["arguments"] = (
                            json.loads(arguments) if arguments.strip() else {}
                        )
                    except json.JSONDecodeError:
                        pass  # 留给模板报错，不静默改数据
            normalized.append(message)
        return normalized

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
        messages = self._normalize_tool_call_arguments(messages)
        rendered = self._tokenizer.apply_chat_template(messages, **kwargs)
        if not isinstance(rendered, str):
            raise TypeError("chat template did not return text")
        return rendered

    # 常见英文词池：合成文本必须 decode→encode 往返稳定。全词表随机采样会产出
    # 多语言杂烩，模型续写的文本 retokenize 后 token 序列漂移（实测 128→132），
    # 多轮 live-echo 下前缀哈希链断裂——radix/存储层前缀复用全部失效。
    _WORD_POOL_TEXT = (
        "the of and to in is that it for as with on be at by this have from "
        "or one had not but what all were when we there can an your which "
        "their said if do will each about how up out them then she many some "
        "so these would other into has more her two like him see time could "
        "no make than first been its who now people my made over did down "
        "only way find use may water long little very after words called "
        "just where most know get through back much before go good new write "
        "our used me man too any day same right look think also around "
        "another came come work three word must because does part even place "
        "well such here take why help put different away again off went old "
        "number great tell men say small every found still between name "
        "should home big give air line set own under read last never us left "
        "end along while might next sound below saw something thought both "
        "few those always show large often together asked house world going "
        "want school important until form food keep children feet land side "
        "without boy once animal life enough took sometimes four head above "
        "kind began almost live page got earth need far hand high year "
        "mother light parts country father let night following picture being "
        "study second eyes soon times story boys since white days ever paper "
        "hard near sentence better best across during today others however "
        "sure means knew tried told young miles sun ways thing whole hear "
        "example heard several change answer room against top turned learn "
        "point city play toward five using himself usually money seen car "
        "morning given order red door sea"
    )
    _word_token_pool: list[int] | None = None

    def _sample_token_ids(self, prompt_length: int) -> list[int]:
        if self._word_token_pool is None:
            pool = []
            for word in dict.fromkeys(self._WORD_POOL_TEXT.split()):
                ids = self._encode(" " + word)
                if len(ids) == 1:
                    pool.append(ids[0])
            if not pool:  # 词表异常时退回原行为
                return self._rng.integers(
                    self._tokenizer.vocab_size, size=prompt_length
                ).tolist()
            self._word_token_pool = pool
        indices = self._rng.integers(len(self._word_token_pool), size=prompt_length)
        return [self._word_token_pool[i] for i in indices]

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
