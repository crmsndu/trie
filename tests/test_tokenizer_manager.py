import json

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from trie.tokenizer_manager import TokenizerManager


def test_tokenizers_backend_fallback_loads_local_artifacts(
    monkeypatch, tmp_path
) -> None:
    tokenizer = Tokenizer(
        WordLevel({"[UNK]": 0, "hello": 1, "world": 2}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "TokenizersBackend",
                "unk_token": "[UNK]",
                "model_max_length": 1024,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "chat_template.jinja").write_text(
        "{% for message in messages %}{{ message.content }}{% endfor %}",
        encoding="utf-8",
    )

    def unsupported_backend(*args, **kwargs):
        raise ValueError(
            "Tokenizer class TokenizersBackend does not exist or is not "
            "currently imported."
        )

    monkeypatch.setattr(
        "trie.tokenizer_manager.AutoTokenizer.from_pretrained",
        unsupported_backend,
    )

    manager = TokenizerManager(str(tmp_path))

    assert manager.encode("hello world") == [1, 2]
    assert manager.render_chat([{"role": "user", "content": "hello"}]) == "hello"


def test_tokenizer_loader_does_not_hide_other_value_errors(monkeypatch) -> None:
    def invalid_tokenizer(*args, **kwargs):
        raise ValueError("different tokenizer error")

    monkeypatch.setattr(
        "trie.tokenizer_manager.AutoTokenizer.from_pretrained",
        invalid_tokenizer,
    )

    with pytest.raises(ValueError, match="different tokenizer error"):
        TokenizerManager("not-a-model")
