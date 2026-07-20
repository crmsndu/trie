import asyncio
from types import SimpleNamespace

from openai.types.completion_choice import CompletionChoice
from openai.types.completion_usage import CompletionUsage, PromptTokensDetails

from trie.client import Client, _extension_token_ids
from trie.results import StreamAccumulator


class ByteTokenizer:
    def count_tokens(self, text: str) -> int:
        return len(text.encode())


class AsyncChunks:
    def __init__(self, chunks: list[object]) -> None:
        self._chunks = iter(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._chunks)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


class FakeCompletions:
    def __init__(self, response: object) -> None:
        self._response = response
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


def _usage() -> CompletionUsage:
    return CompletionUsage(
        completion_tokens=2,
        prompt_tokens=3,
        total_tokens=5,
        prompt_tokens_details=PromptTokensDetails(cached_tokens=1),
    )


def _client(response: object) -> tuple[Client, FakeCompletions]:
    completions = FakeCompletions(response)
    client = Client.__new__(Client)
    client._model = "fake"
    client._client = SimpleNamespace(completions=completions)
    client._tokenizer_manager = ByteTokenizer()
    return client, completions


def test_non_stream_response_reads_vllm_choice_token_ids() -> None:
    choice = CompletionChoice(
        finish_reason="length",
        index=0,
        logprobs=None,
        text="hi",
        prompt_token_ids=[1, 2, 3],
        token_ids=[32, 48],
    )
    response = SimpleNamespace(choices=[choice], usage=_usage(), model_extra={})
    client, completions = _client(response)

    result = asyncio.run(
        client._execute_request(
            "abc",
            2,
            stream=False,
            trace_start=0.0,
            stream_acc=StreamAccumulator(),
            return_token_ids=True,
        )
    )

    assert result.prompt_token_ids == [1, 2, 3]
    assert result.generated_token_ids == [32, 48]
    assert result.cached_tokens == 1
    assert completions.calls[0]["extra_body"]["return_token_ids"] is True


def test_stream_response_accumulates_choice_token_id_deltas() -> None:
    chunks = AsyncChunks(
        [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        text="h",
                        model_extra={
                            "prompt_token_ids": [1, 2, 3],
                            "token_ids": [32],
                        },
                    )
                ],
                usage=None,
                model_extra={},
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        text="i",
                        model_extra={
                            "prompt_token_ids": None,
                            "token_ids": [48],
                        },
                    )
                ],
                usage=None,
                model_extra={},
            ),
            SimpleNamespace(choices=[], usage=_usage(), model_extra={}),
        ]
    )
    client, completions = _client(chunks)

    result = asyncio.run(
        client._execute_request(
            "abc",
            2,
            stream=True,
            trace_start=0.0,
            stream_acc=StreamAccumulator(),
            return_token_ids=True,
        )
    )

    assert result.text == "hi"
    assert result.prompt_token_ids == [1, 2, 3]
    assert result.generated_token_ids == [32, 48]
    assert result.cached_tokens == 1
    assert completions.calls[0]["extra_body"]["return_token_ids"] is True


def test_stream_response_uses_token_ids_when_text_is_empty() -> None:
    chunks = AsyncChunks(
        [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        text="",
                        model_extra={
                            "prompt_token_ids": [1, 2, 3],
                            "token_ids": [32],
                        },
                    )
                ],
                usage=None,
                model_extra={},
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        text="",
                        model_extra={"token_ids": [48]},
                    )
                ],
                usage=None,
                model_extra={},
            ),
            SimpleNamespace(choices=[], usage=_usage(), model_extra={}),
        ]
    )
    client, _ = _client(chunks)

    result = asyncio.run(
        client._execute_request(
            "abc",
            2,
            stream=True,
            trace_start=0.0,
            stream_acc=StreamAccumulator(),
            return_token_ids=True,
        )
    )

    assert result.text == ""
    assert result.generated_token_ids == [32, 48]


def test_non_replay_request_does_not_request_token_ids() -> None:
    choice = CompletionChoice(
        finish_reason="length",
        index=0,
        logprobs=None,
        text="hi",
    )
    response = SimpleNamespace(choices=[choice], usage=_usage(), model_extra={})
    client, completions = _client(response)

    asyncio.run(
        client._execute_request(
            "abc",
            2,
            stream=False,
            trace_start=0.0,
            stream_acc=StreamAccumulator(),
        )
    )

    assert completions.calls[0]["extra_body"] == {"ignore_eos": True}


def test_non_replay_stream_does_not_request_token_ids() -> None:
    chunks = AsyncChunks(
        [
            SimpleNamespace(
                choices=[SimpleNamespace(text="hi", model_extra={})],
                usage=None,
                model_extra={},
            ),
            SimpleNamespace(choices=[], usage=_usage(), model_extra={}),
        ]
    )
    client, completions = _client(chunks)

    asyncio.run(
        client._execute_request(
            "abc",
            2,
            stream=True,
            trace_start=0.0,
            stream_acc=StreamAccumulator(),
        )
    )

    assert completions.calls[0]["extra_body"] == {"ignore_eos": True}


def test_extension_token_ids_supports_model_extra() -> None:
    value = SimpleNamespace(model_extra={"token_ids": [7, 8]})

    assert _extension_token_ids(value, "token_ids") == [7, 8]
