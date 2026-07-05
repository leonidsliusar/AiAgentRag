"""Unit tests for Modal RPC providers."""

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiagentrag.core.exceptions import LLMError
from aiagentrag.core.models import LLMMessage
from aiagentrag.providers.modal.config import (
    ModalRpcConfig,
    ensure_modal_credentials,
    modal_config_from_env,
)
from aiagentrag.providers.modal.embeddings import ModalEmbeddingProvider
from aiagentrag.providers.modal.llm import ModalLLMProvider
from aiagentrag.providers.modal.rpc import ModalRpcClient


@pytest.fixture
def cls_config() -> ModalRpcConfig:
    """Provide a class-based Modal RPC configuration."""
    return ModalRpcConfig(
        app_name="aiagentrag-llm",
        llm_cls="LLMService",
        embed_cls="LLMService",
    )


@pytest.fixture
def function_config() -> ModalRpcConfig:
    """Provide a function-based Modal RPC configuration."""
    return ModalRpcConfig(
        app_name="aiagentrag-llm",
        llm_complete_function="llm_complete",
        llm_stream_function="llm_stream",
        embed_function="embed_texts",
    )


@pytest.fixture
def rpc() -> ModalRpcClient:
    """Provide a Modal RPC client."""
    return ModalRpcClient("aiagentrag-llm")


async def test_modal_llm_complete_via_cls(rpc: ModalRpcClient, cls_config: ModalRpcConfig) -> None:
    """Should complete chat via Modal Cls RPC."""
    provider = ModalLLMProvider(rpc, cls_config)
    with patch.object(rpc, "call_class_method", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = "Modal answer"
        messages = [{"role": "user", "content": "Hi"}]
        result = await provider.complete([LLMMessage(role="user", content="Hi")])
    assert result == "Modal answer"
    mock_call.assert_awaited_once_with("LLMService", "complete", messages)


async def test_modal_llm_stream_via_cls(rpc: ModalRpcClient, cls_config: ModalRpcConfig) -> None:
    """Should stream tokens via Modal Cls generator RPC."""

    async def fake_stream(*_args: object, **_kwargs: object) -> AsyncIterator[str]:
        yield "Hello"
        yield " world"

    provider = ModalLLMProvider(rpc, cls_config)
    with patch.object(rpc, "call_class_method_gen", return_value=fake_stream()):
        tokens = [token async for token in provider.stream([LLMMessage(role="user", content="Hi")])]
    assert tokens == ["Hello", " world"]


async def test_modal_llm_complete_via_function(
    rpc: ModalRpcClient,
    function_config: ModalRpcConfig,
) -> None:
    """Should complete chat via Modal Function RPC."""
    provider = ModalLLMProvider(rpc, function_config)
    with patch.object(rpc, "call_function", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = {"content": "Done"}
        result = await provider.complete([LLMMessage(role="user", content="Hi")])
    assert result == "Done"
    mock_call.assert_awaited_once_with("llm_complete", [{"role": "user", "content": "Hi"}])


async def test_modal_embed_via_cls(rpc: ModalRpcClient, cls_config: ModalRpcConfig) -> None:
    """Should embed texts via Modal Cls RPC."""
    provider = ModalEmbeddingProvider(rpc, cls_config)
    with patch.object(rpc, "call_class_method", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = [[0.1, 0.2], [0.3, 0.4]]
        result = await provider.embed_batch(["a", "b"])
    assert result == [[0.1, 0.2], [0.3, 0.4]]
    mock_call.assert_awaited_once_with("LLMService", "embed", ["a", "b"])


async def test_modal_rpc_client_calls_function_remote() -> None:
    """Should invoke modal.Function.from_name(...).remote()."""
    fake_function = MagicMock()
    fake_function.remote = AsyncMock(return_value="ok")

    fake_modal = MagicMock()
    fake_modal.Function.from_name.return_value = fake_function

    client = ModalRpcClient("app", environment_name="main")
    with patch("aiagentrag.providers.modal.rpc._import_modal", return_value=fake_modal):
        result = await client.call_function("complete", [{"role": "user", "content": "Hi"}])

    assert result == "ok"
    fake_modal.Function.from_name.assert_called_once_with(
        "app",
        "complete",
        environment_name="main",
    )
    fake_function.remote.assert_awaited_once()


async def test_modal_config_from_env_requires_app_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """Should require MODAL_APP_NAME after credentials are present."""
    monkeypatch.setenv("MODAL_TOKEN_ID", "ak-test")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "as-test")
    monkeypatch.delenv("MODAL_APP_NAME", raising=False)

    with pytest.raises(RuntimeError, match="MODAL_APP_NAME"):
        modal_config_from_env()


def test_ensure_modal_credentials_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Should fail fast when neither env vars nor config file provide tokens."""
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)

    fake_config = MagicMock()
    fake_config.get.return_value = None

    with (
        patch("modal.config.config", fake_config),
        pytest.raises(RuntimeError, match="Modal credentials are missing"),
    ):
        ensure_modal_credentials()


def test_ensure_modal_credentials_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Should accept credentials from environment variables."""
    monkeypatch.setenv("MODAL_TOKEN_ID", "ak-test")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "as-test")
    ensure_modal_credentials()


async def test_modal_rpc_missing_sdk_raises_llm_error() -> None:
    """Should raise LLMError when Modal SDK is not installed."""
    client = ModalRpcClient("app")
    missing_modal = LLMError(
        "Modal SDK is not installed. Install with: pip install 'aiagentrag[modal]'",
    )
    with (
        patch("aiagentrag.providers.modal.rpc._import_modal", side_effect=missing_modal),
        pytest.raises(LLMError, match="Modal SDK is not installed"),
    ):
        await client.call_function("complete", [])
