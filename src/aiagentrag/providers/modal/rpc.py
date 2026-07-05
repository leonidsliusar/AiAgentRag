"""Modal RPC invocation helpers."""

from collections.abc import AsyncIterator
from typing import Any

from aiagentrag.core.exceptions import LLMError


def _import_modal() -> Any:
    """Import the Modal SDK or raise a clear error."""
    try:
        import modal
    except ImportError as exc:
        msg = "Modal SDK is not installed. Install with: pip install 'aiagentrag[modal]'"
        raise LLMError(msg) from exc
    return modal


class ModalRpcClient:
    """Call deployed Modal Functions or Cls methods via RPC."""

    def __init__(
        self,
        app_name: str,
        *,
        environment_name: str | None = None,
    ) -> None:
        """Initialize the client for a deployed Modal app."""
        self._app_name = app_name
        self._environment_name = environment_name

    async def call_function(self, function_name: str, *args: Any, **kwargs: Any) -> Any:
        """Invoke a deployed Modal Function and return its result."""
        modal = _import_modal()
        function = modal.Function.from_name(
            self._app_name,
            function_name,
            environment_name=self._environment_name,
        )
        try:
            return await function.remote(*args, **kwargs)
        except Exception as exc:
            msg = f"Modal RPC call to function '{function_name}' failed: {exc}"
            raise LLMError(msg) from exc

    async def call_function_gen(
        self,
        function_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Invoke a deployed Modal generator Function and yield its outputs."""
        modal = _import_modal()
        function = modal.Function.from_name(
            self._app_name,
            function_name,
            environment_name=self._environment_name,
        )
        try:
            async for item in function.remote_gen(*args, **kwargs):
                yield item
        except Exception as exc:
            msg = f"Modal RPC stream to function '{function_name}' failed: {exc}"
            raise LLMError(msg) from exc

    async def call_class_method(
        self,
        cls_name: str,
        method_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Invoke a method on a deployed Modal Cls instance."""
        modal = _import_modal()
        cls = modal.Cls.from_name(
            self._app_name,
            cls_name,
            environment_name=self._environment_name,
        )
        method = getattr(cls(), method_name)
        try:
            return await method.remote(*args, **kwargs)
        except Exception as exc:
            msg = f"Modal RPC call to {cls_name}.{method_name} failed: {exc}"
            raise LLMError(msg) from exc

    async def call_class_method_gen(
        self,
        cls_name: str,
        method_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Invoke a generator method on a deployed Modal Cls instance."""
        modal = _import_modal()
        cls = modal.Cls.from_name(
            self._app_name,
            cls_name,
            environment_name=self._environment_name,
        )
        method = getattr(cls(), method_name)
        try:
            async for item in method.remote_gen(*args, **kwargs):
                yield item
        except Exception as exc:
            msg = f"Modal RPC stream to {cls_name}.{method_name} failed: {exc}"
            raise LLMError(msg) from exc
