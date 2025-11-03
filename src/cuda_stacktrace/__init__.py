"""
User-friendly Python API for CUDA stack tracing via CUPTI.

This package wraps the native extension (cuda_stacktrace._native) and exposes:

- enable(api_names, domains=("runtime",), site="enter", only_current_thread=False, thread_idents=None, once_per_line=False)
- disable() / is_enabled()
- set_functions(api_names)
- CudaStackTracer context manager for scoped tracing

Notes:
- stream: If provided, stderr is temporarily redirected to the given stream
  within the context manager, so native output is captured there.
"""

from __future__ import annotations

from contextlib import redirect_stderr
from typing import Iterable, Optional, Union, Iterable as _Iterable
import warnings

try:
    from . import _native as _ext
except Exception as e:  # pragma: no cover - import-time failure
    raise RuntimeError(
        "Failed to import native cuda_stacktrace extension; ensure the package is built and installed."
    ) from e


def enable(
    api_names: Iterable[str],
    *,
    domains: Iterable[str] | None = ("runtime",),
    site: str = "enter",
    only_current_thread: bool = False,
    thread_idents: Optional[_Iterable[int]] = None,
    once_per_line: bool = False,
) -> None:
    """Enable stack printing for selected CUDA API names.

    - api_names: iterable of CUDA API names to watch (exact names)
    - domains: ("runtime",), ("driver",), or ("runtime", "driver")
    - site: "enter" or "exit"
    - only_current_thread: if True, only log for the thread that called enable()
    - thread_idents: iterable of Python thread idents (ints) to allow
    - once_per_line: if True, print only the first stack for each originating
      Python callsite (filename:lineno of the bottommost frame)
    """
    _ext.enable(
        api_names,
        domains=tuple(domains) if domains is not None else None,
        site=site,
        only_current_thread=bool(only_current_thread),
        thread_idents=list(thread_idents) if thread_idents is not None else None,
        once_per_line=bool(once_per_line),
    )


def disable() -> None:
    """Disable stack printing and detach CUPTI."""
    _ext.disable()


def is_enabled() -> bool:
    """Return whether logging is currently enabled."""
    return bool(_ext.is_enabled())


def set_functions(functions: Iterable[str]) -> None:
    """Update the allow-list of CUDA API names without changing enabled state."""
    _ext.set_functions(functions)


def start(
    api_names: Iterable[str],
    *,
    domains: Iterable[str] | None = ("runtime",),
    site: str = "enter",
    only_current_thread: bool = False,
    thread_idents: Optional[_Iterable[int]] = None,
    once_per_line: bool = False,
) -> None:
    """Alias for enable(api_names, domains=..., site=...)."""
    enable(
        api_names,
        domains=domains,
        site=site,
        only_current_thread=only_current_thread,
        thread_idents=thread_idents,
        once_per_line=once_per_line,
    )


def stop() -> None:
    """Alias for disable()."""
    disable()


class CudaStackTracer:
    """Context manager for scoped CUDA stack tracing.

    Example:
        with CudaStackTracer(functions=["cudaMalloc", "cuStreamSynchronize"], enabled=True):
            ... do CUDA work ...
    """

    def __init__(
        self,
        *,
        functions: Optional[Union[str, Iterable[str]]] = None,
        enabled: bool = True,
        only_current_thread: Optional[bool] = None,
        local_thread_only: Optional[bool] = None,
        domains: Iterable[str] | None = ("runtime",),
        site: str = "enter",
        stream=None,
        once_per_line: bool = False,
    ) -> None:
        self.functions = (
            [functions]
            if isinstance(functions, str)
            else list(functions) if functions is not None else []
        )
        self.enabled = bool(enabled)
        # Prefer only_current_thread; keep local_thread_only as deprecated alias
        if only_current_thread is None and local_thread_only is not None:
            warnings.warn(
                "'local_thread_only' is deprecated; use 'only_current_thread'",
                DeprecationWarning,
                stacklevel=2,
            )
            only_current_thread = bool(local_thread_only)
        self.only_current_thread = (
            bool(only_current_thread) if only_current_thread is not None else False
        )
        self.domains = tuple(domains) if domains is not None else None
        self.site = site
        self.stream = stream
        self.once_per_line = bool(once_per_line)
        self._prev_enabled: Optional[bool] = None
        self._redir_cm = None

    def __enter__(self):
        self._prev_enabled = is_enabled()

        if self.functions:
            set_functions(self.functions)

        if self.enabled:
            try:
                enable(
                    self.functions,
                    domains=self.domains,
                    site=self.site,
                    only_current_thread=self.only_current_thread,
                    once_per_line=self.once_per_line,
                )
            except RuntimeError:
                # Re-raise to make failures explicit within context usage
                raise

        if self.stream is not None:
            self._redir_cm = redirect_stderr(self.stream)
            self._redir_cm.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._redir_cm is not None:
            self._redir_cm.__exit__(exc_type, exc, tb)
            self._redir_cm = None

        # Only disable if we enabled in __enter__ and it was previously disabled
        if self.enabled and self._prev_enabled is False:
            disable()
        # If it was previously enabled, we leave it as-is
        return False
