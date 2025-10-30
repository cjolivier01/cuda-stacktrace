import os
import glob
import ctypes
import sys
import pytest

# Try regular import; if it fails, attempt to import from local build/lib path.
try:
    import cuda_stacktrace as cst  # type: ignore
except Exception as e:  # pragma: no cover - test env fallback
    # Try to locate build/lib.* directory and add to sys.path
    build_dirs = []
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # Ensure Python can find the package sources
    src_dir = os.path.join(root, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    # And also the compiled extension under build/lib.*
    cand = os.path.join(root, "build")
    if os.path.isdir(cand):
        for name in os.listdir(cand):
            if name.startswith("lib."):
                build_dirs.append(os.path.join(cand, name))
    for p in build_dirs:
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        import cuda_stacktrace as cst  # type: ignore
    except Exception as e2:
        pytest.skip(f"cuda_stacktrace import failed: {e2}", allow_module_level=True)


def have_lib(name_candidates):
    # Try by soname via loader search paths
    for name in name_candidates:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    # Try absolute paths in common CUDA lib dir
    libdir = "/usr/lib/x86_64-linux-gnu"
    for name in name_candidates:
        base = os.path.join(libdir, name)
        for cand in [base] + glob.glob(base + "*"):
            try:
                return ctypes.CDLL(cand)
            except OSError:
                continue
    return None


def test_import_and_toggle():
    assert cst.is_enabled() is False
    try:
        cst.enable(["cudaMalloc"])  # default domain runtime, site enter
    except RuntimeError as e:
        pytest.skip(f"CUPTI not available/working here: {e}")
    assert cst.is_enabled() is True
    cst.disable()
    assert cst.is_enabled() is False


def test_import_only():
    assert hasattr(cst, "enable")
    assert cst.is_enabled() is False


@pytest.mark.parametrize("site", ["enter", "exit"])
def test_capture_cudaMalloc(capfd, site):
    try:
        cst.enable(["cudaMalloc"], domains=("runtime",), site=site)
    except RuntimeError as e:
        pytest.skip(f"CUPTI not available/working here: {e}")

    libcudart = have_lib([
        "libcudart.so",
        "libcudart.so.12",
        "libcudart.so.11.0",
        "libcudart.so.10.2",
    ])
    if libcudart is None:
        cst.disable()
        pytest.skip("libcudart not found on system")

    # Prepare signature for cudaMalloc
    cudaMalloc = libcudart.cudaMalloc
    cudaMalloc.argtypes = (ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t)
    cudaMalloc.restype = ctypes.c_int

    # Call it with tiny size; return code may be non-zero on machines without GPUs
    ptr = ctypes.c_void_p()
    _ = cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(4))

    cst.disable()

    captured = capfd.readouterr()
    # We print to stderr from the extension
    err = captured.err
    assert "[cuda_stacktrace]" in err
    assert "cudaMalloc" in err
    # The python stack should include this test function name
    assert "test_capture_cudaMalloc" in err


def test_filtering_blocks_output(capfd):
    try:
        # Set filter to a non-existent API so that cudaMalloc doesn't match.
        cst.enable(["DefinitelyNotAnApi"], domains=("runtime",), site="enter")
    except RuntimeError as e:
        pytest.skip(f"CUPTI not available/working here: {e}")

    libcudart = have_lib([
        "libcudart.so",
        "libcudart.so.12",
        "libcudart.so.11.0",
        "libcudart.so.10.2",
    ])
    if libcudart is None:
        cst.disable()
        pytest.skip("libcudart not found on system")

    cudaMalloc = libcudart.cudaMalloc
    cudaMalloc.argtypes = (ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t)
    cudaMalloc.restype = ctypes.c_int

    ptr = ctypes.c_void_p()
    _ = cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(4))

    cst.disable()

    captured = capfd.readouterr()
    assert "cudaMalloc" not in captured.err


def test_context_manager_scoped_enable(capfd):
    """Use the friendly context manager API."""
    # Ensure disabled to start
    if cst.is_enabled():
        cst.disable()

    # Will skip if CUPTI not available when enabling
    libcudart = have_lib([
        "libcudart.so",
        "libcudart.so.12",
        "libcudart.so.11.0",
        "libcudart.so.10.2",
    ])
    if libcudart is None:
        pytest.skip("libcudart not found on system")

    # Prepare signature for cudaMalloc
    cudaMalloc = libcudart.cudaMalloc
    cudaMalloc.argtypes = (ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t)
    cudaMalloc.restype = ctypes.c_int

    # Within context, enabling should produce output
    try:
        with cst.CudaStackTracer(functions=["cudaMalloc"], enabled=True, only_current_thread=True):
            ptr = ctypes.c_void_p()
            _ = cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(4))
    except RuntimeError as e:
        pytest.skip(f"CUPTI not available/working here: {e}")

    captured = capfd.readouterr()
    assert "[cuda_stacktrace]" in captured.err
    assert "cudaMalloc" in captured.err

    # After context, it should be disabled again
    assert cst.is_enabled() is False
