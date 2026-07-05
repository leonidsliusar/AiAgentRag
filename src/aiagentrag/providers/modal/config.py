"""Configuration for Modal RPC targets."""

from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator


class ModalRpcConfig(BaseModel):
    """Resolved Modal app and RPC target names."""

    model_config = ConfigDict(frozen=True)

    app_name: str
    environment_name: str | None = None

    llm_cls: str | None = None
    llm_complete_method: str = "complete"
    llm_stream_method: str = "stream"
    llm_complete_function: str | None = None
    llm_stream_function: str | None = None

    embed_cls: str | None = None
    embed_method: str = "embed"
    embed_function: str | None = None

    @model_validator(mode="after")
    def validate_llm_target(self) -> Self:
        """Ensure LLM RPC target is configured exactly one way."""
        cls_mode = self.llm_cls is not None
        fn_mode = self.llm_complete_function is not None or self.llm_stream_function is not None
        if cls_mode and fn_mode:
            msg = "Configure either llm_cls or llm_*_function, not both"
            raise ValueError(msg)
        if not cls_mode and not fn_mode:
            msg = "Configure llm_cls or llm_complete_function and llm_stream_function"
            raise ValueError(msg)
        if fn_mode and (not self.llm_complete_function or not self.llm_stream_function):
            msg = "Both llm_complete_function and llm_stream_function are required"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_embed_target(self) -> Self:
        """Ensure embedding RPC target is configured exactly one way."""
        cls_mode = self.embed_cls is not None
        fn_mode = self.embed_function is not None
        if cls_mode and fn_mode:
            msg = "Configure either embed_cls or embed_function, not both"
            raise ValueError(msg)
        if not cls_mode and not fn_mode:
            msg = "Configure embed_cls or embed_function"
            raise ValueError(msg)
        return self

    @property
    def uses_llm_cls(self) -> bool:
        """Return True when LLM calls go through a Modal class."""
        return self.llm_cls is not None

    @property
    def uses_embed_cls(self) -> bool:
        """Return True when embedding calls go through a Modal class."""
        return self.embed_cls is not None


def modal_config_from_env() -> ModalRpcConfig:
    """Build Modal RPC config from standard environment variables."""
    import os

    app_name = os.environ.get("MODAL_APP_NAME")
    if not app_name:
        msg = "MODAL_APP_NAME is required for Modal provider"
        raise RuntimeError(msg)

    llm_cls = os.environ.get("MODAL_LLM_CLS")
    llm_complete_fn = os.environ.get("MODAL_LLM_COMPLETE_FUNCTION")
    llm_stream_fn = os.environ.get("MODAL_LLM_STREAM_FUNCTION")

    embed_cls = os.environ.get("MODAL_EMBED_CLS")
    if embed_cls is None and llm_cls:
        embed_cls = llm_cls
    embed_fn = os.environ.get("MODAL_EMBED_FUNCTION")

    return ModalRpcConfig(
        app_name=app_name,
        environment_name=os.environ.get("MODAL_ENVIRONMENT"),
        llm_cls=llm_cls,
        llm_complete_method=os.environ.get("MODAL_LLM_COMPLETE_METHOD", "complete"),
        llm_stream_method=os.environ.get("MODAL_LLM_STREAM_METHOD", "stream"),
        llm_complete_function=llm_complete_fn,
        llm_stream_function=llm_stream_fn,
        embed_cls=embed_cls,
        embed_method=os.environ.get("MODAL_EMBED_METHOD", "embed"),
        embed_function=embed_fn,
    )
