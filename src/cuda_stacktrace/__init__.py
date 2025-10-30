"""
User-friendly Python API for CUDA stack tracing via CUPTI.

This package wraps the native extension (cuda_stacktrace._native) and exposes:

- enable(functions, domains=("runtime",), site="enter") / disable() / is_enabled()
- set_functions(functions)
- CudaStackTracer context manager for scoped tracing

Notes:
- local_thread_only: Currently the extension only captures the stack of the
  OS thread invoking the CUDA API (i.e., local thread). This parameter is
  accepted for forward compatibility and to mirror desired API shape.
- stream: If provided, stderr is temporarily redirected to the given stream
  within the context manager, so native output is captured there.
"""

from __future__ import annotations

from contextlib import contextmanager, redirect_stderr
from typing import Iterable, Optional

try:
    from . import _native as _ext
except Exception as e:  # pragma: no cover - import-time failure
    raise RuntimeError(
        "Failed to import native cuda_stacktrace extension; ensure the package is built and installed."
    ) from e


def enable(
    functions: Iterable[str],
    *,
    domains: Iterable[str] | None = ("runtime",),
    site: str = "enter",
) -> None:
    """Enable stack printing for selected CUDA API names.

    - functions: iterable of CUDA API names to watch (exact names)
    - domains: ("runtime",), ("driver",), or ("runtime", "driver")
    - site: "enter" or "exit"
    """
    _ext.enable(functions, domains=tuple(domains) if domains is not None else None, site=site)


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
    functions: Iterable[str],
    *,
    domains: Iterable[str] | None = ("runtime",),
    site: str = "enter",
) -> None:
    """Alias for enable(functions, domains=..., site=...)."""
    enable(functions, domains=domains, site=site)


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
        functions: Optional[Iterable[str]] = None,
        enabled: bool = True,
        local_thread_only: bool = True,
        domains: Iterable[str] | None = ("runtime",),
        site: str = "enter",
        stream=None,
    ) -> None:
        self.functions = list(functions) if functions is not None else []
        self.enabled = bool(enabled)
        self.local_thread_only = bool(local_thread_only)  # currently only mode supported
        self.domains = tuple(domains) if domains is not None else None
        self.site = site
        self.stream = stream
        self._prev_enabled: Optional[bool] = None
        self._redir_cm = None

    def __enter__(self):
        self._prev_enabled = is_enabled()

        if self.functions:
            set_functions(self.functions)

        if self.enabled:
            try:
                enable(self.functions, domains=self.domains, site=self.site)
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

