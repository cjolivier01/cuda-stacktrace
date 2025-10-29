import os
import sys
import glob
import ctypes
import importlib.util
import pytest


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


def import_cst():
    try:
        import cuda_stacktrace as m  # type: ignore
        return m
    except Exception:
        # Fallback: load from local built .so
        candidates = glob.glob(os.path.join(os.path.dirname(__file__), "..", "cuda_stacktrace*.so"))
        candidates += glob.glob(os.path.join(os.getcwd(), "cuda_stacktrace*.so"))
        candidates = [os.path.abspath(p) for p in candidates]
        if not candidates:
            raise
        path = candidates[0]
        spec = importlib.util.spec_from_file_location("cuda_stacktrace", path)
        assert spec and spec.loader
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)  # type: ignore
        return m


def test_import_and_toggle():
    try:
        cst = import_cst()
    except Exception as e:
        pytest.skip(f"cuda_stacktrace import failed: {e}")

    assert cst.is_enabled() is False
    try:
        cst.enable(["cudaMalloc"])  # default domain runtime, site enter
    except RuntimeError as e:
        pytest.skip(f"CUPTI not available/working here: {e}")
    assert cst.is_enabled() is True
    cst.disable()
    assert cst.is_enabled() is False


def test_import_only():
    m = import_cst()
    assert hasattr(m, "enable")
    assert m.is_enabled() is False


@pytest.mark.parametrize("site", ["enter", "exit"])
def test_capture_cudaMalloc(capfd, site):
    try:
        cst = import_cst()
    except Exception as e:
        pytest.skip(f"cuda_stacktrace import failed: {e}")

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
        cst = import_cst()
    except Exception as e:
        pytest.skip(f"cuda_stacktrace import failed: {e}")

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
