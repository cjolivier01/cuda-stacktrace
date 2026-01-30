"""
@file __init__.py
@brief User-friendly Python API for CUDA stack tracing via CUPTI.

This package wraps the native extension :mod:`cuda_stacktrace._native` and
exposes:

- :func:`enable` / :func:`start` to turn tracing on
- :func:`disable` / :func:`stop` to turn tracing off
- :func:`set_functions` to update the CUDA API allow-list
- :func:`is_enabled` to query current state
- :class:`CudaStackTracer` context manager for scoped tracing

Notes:
- If ``stream`` is provided to :class:`CudaStackTracer`, @c stderr is
  temporarily redirected to the given stream within the context manager, so
  native output is captured there.
"""

from __future__ import annotations

import ctypes
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
    """@brief Enable stack printing for selected CUDA API names.

    @param api_names Iterable of CUDA API names to watch (exact names).
    @param domains Iterable of domains: ``("runtime",)``, ``("driver",)``,
                   or ``("runtime", "driver")``. If @c None, defaults to runtime.
    @param site Callback site to trace: ``"enter"`` or ``"exit"``.
    @param only_current_thread If @c True, only log for the thread that called
                               :func:`enable`.
    @param thread_idents Iterable of Python thread identifiers (ints) to allow.
                         Logging occurs only on those threads when provided.
    @param once_per_line If @c True, print only the first stack for each
                         originating Python callsite (filename:lineno of the
                         bottommost frame).
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
    """@brief Disable stack printing and detach CUPTI."""
    _ext.disable()


def is_enabled() -> bool:
    """@brief Query whether logging is currently enabled.

    @return @c True if tracing is enabled; @c False otherwise.
    """
    return bool(_ext.is_enabled())


def set_functions(functions: Iterable[str]) -> None:
    """@brief Update the allow-list of CUDA API names.

    This does not change whether tracing is currently enabled or disabled.

    @param functions Iterable of CUDA API names (exact strings) to watch.
    """
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
    """@brief Alias for :func:`enable`.

    This is a convenience wrapper that directly forwards arguments to
    :func:`enable`.
    """
    enable(
        api_names,
        domains=domains,
        site=site,
        only_current_thread=only_current_thread,
        thread_idents=thread_idents,
        once_per_line=once_per_line,
    )


def stop() -> None:
    """@brief Alias for :func:`disable`."""
    disable()


class CudaStackTracer:
    """@brief Context manager for scoped CUDA stack tracing.

    Example:
    @code{.py}
        from cuda_stacktrace import CudaStackTracer

        with CudaStackTracer(functions=["cudaMalloc", "cuStreamSynchronize"], enabled=True):
            # ... do CUDA work ...
            pass
    @endcode
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
        """@brief Construct a new :class:`CudaStackTracer`.

        @param functions Single CUDA API name or iterable of names to trace.
                         If @c None or empty, all APIs in the selected domains
                         are watched.
        @param enabled If @c True, tracing is enabled upon entering the context.
        @param only_current_thread If @c True, restrict tracing to the thread
                                   that enters the context.
        @param local_thread_only Deprecated alias for @p only_current_thread.
        @param domains Iterable of domains: ``("runtime",)``, ``("driver",)``,
                       or both. If @c None, defaults to runtime.
        @param site Callback site: ``"enter"`` or ``"exit"``.
        @param stream Optional file-like object used to temporarily redirect
                      @c stderr while the context is active.
        @param once_per_line If @c True, enable once-per-line deduplication
                             based on Python callsite.
        """
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
        self._warned_missing = False

    @staticmethod
    def _load_first_library(candidates):
        for name in candidates:
            if not name:
                continue
            try:
                return ctypes.CDLL(name)
            except OSError:
                continue
        return None

    def _warn_on_missing_functions(self):
        if self._warned_missing or not self.functions:
            return

        # Default to runtime domain when not specified
        domain_set = set(self.domains or ("runtime",))

        libs = []
        if "runtime" in domain_set:
            libs.append(
                self._load_first_library(
                    [
                        "libcudart.so",
                        "libcudart.so.13",
                        "libcudart.so.12",
                        "libcudart.so.11.0",
                        "libcudart.so.10.2",
                    ]
                )
            )
        if "driver" in domain_set:
            libs.append(
                self._load_first_library(
                    [
                        "libcuda.so",
                        "libcuda.so.1",
                    ]
                )
            )

        libs = [lib for lib in libs if lib is not None]
        if not libs:
            return  # Cannot validate without a library

        missing = []
        for fn in self.functions:
            found = False
            for lib in libs:
                try:
                    getattr(lib, fn)
                    found = True
                    break
                except AttributeError:
                    continue
            if not found:
                missing.append(fn)

        if missing:
            self._warned_missing = True
            warnings.warn(
                f"CUDA APIs not found in loaded libraries: {', '.join(sorted(set(missing)))}",
                RuntimeWarning,
                stacklevel=3,
            )

    def __enter__(self):
        """@brief Enter the tracing context.

        Saves the previous enabled state, optionally updates the function
        allow-list, and enables tracing according to the instance settings.

        @return Self, so the context manager can be bound if desired.
        """
        self._prev_enabled = is_enabled()

        if self.functions:
            set_functions(self.functions)
            self._warn_on_missing_functions()

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
        """@brief Exit the tracing context.

        Restores any temporary stderr redirection and disables tracing again
        if it was previously disabled on entry and this instance enabled it.

        @param exc_type Exception type, if any.
        @param exc Exception instance, if any.
        @param tb Traceback object, if any.
        @return Always @c False to propagate exceptions.
        """
        if self._redir_cm is not None:
            self._redir_cm.__exit__(exc_type, exc, tb)
            self._redir_cm = None

        # Only disable if we enabled in __enter__ and it was previously disabled
        if self.enabled and self._prev_enabled is False:
            disable()
        # If it was previously enabled, we leave it as-is
        return False
