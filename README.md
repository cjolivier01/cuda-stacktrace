cuda-stacktrace
================

Print the Python call stack whenever selected CUDA APIs are called, using CUPTI callbacks. Useful for finding where CUDA runtime/driver calls originate in Python programs.

Features
- Watch specific APIs (e.g., `cudaMalloc`, `cudaMemcpy`, `cudaLaunchKernel`) or all in a domain.
- Domains: `"runtime"` and/or `"driver"`.
- Callback site: `"enter"` or `"exit"`.
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
Minimal example tracing `cudaMalloc` on runtime API entry:

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

Output appears on `stderr` starting with:

```
[cuda_stacktrace] runtime/enter cudaMalloc
<python stack frames here>
```

You can change domains and site:

```python
# Driver API examples: cst.enable(["cuMemAlloc", "cuLaunchKernel"], domains=("driver",), site="exit")
# Watch everything in runtime domain: cst.enable([], domains=("runtime",))
```

Python API
- `enable(api_names, domains=("runtime",), site="enter")`
- `disable()`
- `set_functions(api_names)`
- `is_enabled() -> bool`

Notes:
- If `api_names` is empty, all APIs in the chosen domain(s) are logged.
- Output is written to `stderr`.
- If you see `CUPTI_ERROR_NOT_INITIALIZED`, ensure the CUDA driver is initialized (`nvidia-smi` should work) and `libcupti` is in your library path.

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

