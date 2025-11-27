cuda-stacktrace
================

Print the Python call stack whenever selected CUDA APIs are called, using CUPTI callbacks. Useful for finding where CUDA runtime/driver calls originate in Python programs.

Features
- Watch specific APIs (e.g., `cudaMalloc`, `cudaMemcpy`, `cudaLaunchKernel`) or all in a domain.
- Domains: `"runtime"` and/or `"driver"`.
- Callback site: `"enter"` or `"exit"`.
- Optional thread filtering: only report for the thread that enabled tracing or a list of thread idents.
- Optional once-per-line deduplication by originating Python callsite.
- Enable/disable from Python at runtime.
- Output goes to `stderr` with a clear prefix.

Requirements
- Python 3.8+
- C++17-capable compiler (e.g., GCC 9+)
- CUPTI runtime library (`libcupti`) available at runtime
- NVIDIA driver and CUDA install for functional tracing

On Debian/Ubuntu-like systems:
- `libcupti` is commonly found in `/usr/lib/x86_64-linux-gnu`.
- Ensure it’s on the loader path, e.g. `export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH`.

Build and Install
- Build in-place for development:
  - `python3 setup.py build_ext --inplace`
- Or install as a package:
  - `pip install .`

Usage

### Recommended: context manager (`CudaStackTracer`)

Use the high-level context manager to scope tracing to a block of code. This is
often the simplest way to track where CUDA calls originate:

```python
import ctypes
from cuda_stacktrace import CudaStackTracer

# Call a CUDA runtime function via ctypes; errors are okay for stack printing
libcudart = ctypes.CDLL("libcudart.so")  # try lib names like libcudart.so.12/.13 if needed
libcudart.cudaMalloc.argtypes = (ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t)
libcudart.cudaMalloc.restype = ctypes.c_int

with CudaStackTracer(functions=["cudaMalloc"], enabled=True, only_current_thread=True):
    ptr = ctypes.c_void_p()
    rc = libcudart.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(4))
```

Output appears on `stderr` starting with something like:

```
[cuda_stacktrace] runtime/enter cudaMalloc
<python stack frames here>
```

`CudaStackTracer` parameters:
- `functions`: string or iterable of CUDA API names to watch (e.g., `["cudaMalloc"]`).
- `domains`: `("runtime",)`, `("driver",)`, or `("runtime", "driver")`.
- `site`: `"enter"` or `"exit"`.
- `enabled`: whether to enable tracing inside the context.
- `only_current_thread`: restrict logging to the thread that entered the context.
- `stream`: optional file-like object to temporarily capture `stderr` output.
- `once_per_line`: if `True`, only print the first stack per Python callsite.

When the context exits, tracing is automatically disabled again if it was
previously disabled on entry.

### Lower-level API (`enable` / `disable`)

You can also control tracing globally using the functional API:

```python
import ctypes
import cuda_stacktrace as cst

# Enable for specific function names, runtime domain, at entry
cst.enable(["cudaMalloc"], domains=("runtime",), site="enter")

# Call a CUDA runtime function via ctypes; errors are okay for stack printing
libcudart = ctypes.CDLL("libcudart.so")  # try lib names like libcudart.so.12/.13 if needed
libcudart.cudaMalloc.argtypes = (ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t)
libcudart.cudaMalloc.restype = ctypes.c_int
ptr = ctypes.c_void_p()
rc = libcudart.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(4))

cst.disable()
```

You can change domains and site:

```python
# Driver API examples: cst.enable(["cuMemAlloc", "cuLaunchKernel"], domains=("driver",), site="exit")
# Watch everything in runtime domain: cst.enable([], domains=("runtime",))
```

Python API
- `enable(api_names, domains=("runtime",), site="enter", only_current_thread=False, thread_idents=None)`
- `start(api_names, domains=("runtime",), site="enter", only_current_thread=False, thread_idents=None)` (alias for `enable`)
- `disable()`
- `stop()` (alias for `disable`)
- `set_functions(api_names)`
- `is_enabled() -> bool`
- `CudaStackTracer(...)` context manager for scoped tracing

Thread filtering
- `only_current_thread=True` restricts logging to the Python thread that calls `enable(...)`.
- `thread_idents` accepts an iterable of integers matching `threading.get_ident()` values; logging occurs only on those threads.
- If both are provided, a callback is logged if it matches either condition.

Notes:
- If `api_names` is empty, all APIs in the chosen domain(s) are logged.
- Output is written to `stderr`.
- If you see `CUPTI_ERROR_NOT_INITIALIZED`, ensure the CUDA driver is initialized (`nvidia-smi` should work) and `libcupti` is in your library path.

Limitations
- Stream-based filtering is not supported in this build. Determining the CUDA stream for arbitrary API calls via CUPTI callbacks requires decoding per-API parameter structures, which this extension intentionally avoids to keep build-time dependencies low and maintain compatibility across CUDA versions. If you need stream filtering, please open an issue to discuss enabling a CUDA/CUPTI header–based build that can parse function parameters for common async APIs.

Tests
- Run tests (they will skip gracefully when CUPTI isn’t available):
  - `pytest -q`
- On systems where `libcupti` is under `/usr/lib/x86_64-linux-gnu`, set:
  - `export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH`
  - `pytest -q -s`

Implementation Notes
- Implemented as a CPython C extension; no pybind11 dependency.
- Uses CUPTI callback API and only accesses the `callbackSite` and `functionName` fields for compatibility.
- Acquires the Python GIL in the callback and formats the stack via `traceback.format_stack`.
- A small reentrancy guard prevents recursive callbacks during printing.
