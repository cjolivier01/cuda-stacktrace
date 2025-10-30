import ctypes
import glob
import os
import threading
import time
import pytest
import sys


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
        # Fallback 1: add local src + build/lib.* to sys.path and try package import
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        src_dir = os.path.join(root, "src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        cand_build = os.path.join(root, "build")
        build_dirs = []
        if os.path.isdir(cand_build):
            for name in os.listdir(cand_build):
                if name.startswith("lib."):
                    build_dirs.append(os.path.join(cand_build, name))
        for p in build_dirs:
            if p not in sys.path:
                sys.path.insert(0, p)
        try:
            import cuda_stacktrace as m2  # type: ignore
            return m2
        except Exception:
            # Fallback 2: load from a top-level built .so if present
            candidates = glob.glob(os.path.join(os.path.dirname(__file__), "..", "cuda_stacktrace*.so"))
            candidates += glob.glob(os.path.join(os.getcwd(), "cuda_stacktrace*.so"))
            candidates = [os.path.abspath(p) for p in candidates]
            if not candidates:
                raise
            path = candidates[0]
            import importlib.util

            spec = importlib.util.spec_from_file_location("cuda_stacktrace", path)
            assert spec and spec.loader
            m3 = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m3)  # type: ignore
            return m3


def get_libcudart_or_skip():
    libcudart = have_lib([
        "libcudart.so",
        "libcudart.so.12",
        "libcudart.so.11.0",
        "libcudart.so.10.2",
    ])
    if libcudart is None:
        pytest.skip("libcudart not found on system")
    return libcudart


def setup_runtime_funcs(libcudart):
    # cudaMalloc
    libcudart.cudaMalloc.argtypes = (ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t)
    libcudart.cudaMalloc.restype = ctypes.c_int

    # cudaStreamCreate
    libcudart.cudaStreamCreate.argtypes = (ctypes.POINTER(ctypes.c_void_p),)
    libcudart.cudaStreamCreate.restype = ctypes.c_int

    # cudaStreamSynchronize
    libcudart.cudaStreamSynchronize.argtypes = (ctypes.c_void_p,)
    libcudart.cudaStreamSynchronize.restype = ctypes.c_int

    # cudaStreamDestroy
    libcudart.cudaStreamDestroy.argtypes = (ctypes.c_void_p,)
    libcudart.cudaStreamDestroy.restype = ctypes.c_int


def do_malloc(libcudart):
    ptr = ctypes.c_void_p()
    _ = libcudart.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(4))
    return ptr


def do_stream_ops(libcudart):
    stream = ctypes.c_void_p()
    _ = libcudart.cudaStreamCreate(ctypes.byref(stream))
    try:
        _ = libcudart.cudaStreamSynchronize(stream)
    finally:
        _ = libcudart.cudaStreamDestroy(stream)


def test_only_current_thread_filters_other_threads_with_malloc(capfd):
    try:
        cst = import_cst()
    except Exception as e:
        pytest.skip(f"cuda_stacktrace import failed: {e}")

    libcudart = get_libcudart_or_skip()
    setup_runtime_funcs(libcudart)

    try:
        cst.enable(["cudaMalloc"], domains=("runtime",), site="enter", only_current_thread=True)
    except RuntimeError as e:
        pytest.skip(f"CUPTI not available/working here: {e}")

    # Call from main thread: should produce output
    def main_malloc_marker():
        do_malloc(libcudart)

    main_malloc_marker()
    out1 = capfd.readouterr().err
    cst.disable()
    assert "[cuda_stacktrace]" in out1 and "cudaMalloc" in out1 and "main_malloc_marker" in out1

    # Re-enable and test worker thread blocked by filter
    try:
        cst.enable(["cudaMalloc"], domains=("runtime",), site="enter", only_current_thread=True)
    except RuntimeError as e:
        pytest.skip(f"CUPTI not available/working here: {e}")

    def worker_malloc_marker():
        do_malloc(libcudart)

    capfd.readouterr()  # clear
    t = threading.Thread(target=worker_malloc_marker, name="cst-worker-1")
    t.start()
    t.join()
    out2 = capfd.readouterr().err
    cst.disable()
    assert "cudaMalloc" not in out2


def test_thread_idents_allows_only_worker(capfd):
    try:
        cst = import_cst()
    except Exception as e:
        pytest.skip(f"cuda_stacktrace import failed: {e}")

    libcudart = get_libcudart_or_skip()
    setup_runtime_funcs(libcudart)

    worker_ident_holder = {}
    ready = threading.Event()
    go = threading.Event()

    def worker_malloc_marker():
        worker_ident_holder["ident"] = threading.get_ident()
        ready.set()
        go.wait(5)
        do_malloc(libcudart)

    t = threading.Thread(target=worker_malloc_marker, name="cst-worker-2")
    t.start()
    assert ready.wait(5)
    wid = worker_ident_holder.get("ident")
    assert wid is not None

    try:
        cst.enable(["cudaMalloc"], domains=("runtime",), site="enter", thread_idents=[int(wid)])
    except RuntimeError as e:
        go.set()
        t.join()
        pytest.skip(f"CUPTI not available/working here: {e}")

    # Call from main thread should NOT be logged
    do_malloc(libcudart)
    out_main = capfd.readouterr().err
    assert "cudaMalloc" not in out_main

    # Allow worker to run; its call should be logged
    capfd.readouterr()  # clear
    go.set()
    t.join()
    out_worker = capfd.readouterr().err
    cst.disable()
    assert "[cuda_stacktrace]" in out_worker and "cudaMalloc" in out_worker and "worker_malloc_marker" in out_worker


def test_stream_ops_only_current_thread_filters_other_threads(capfd):
    try:
        cst = import_cst()
    except Exception as e:
        pytest.skip(f"cuda_stacktrace import failed: {e}")

    libcudart = get_libcudart_or_skip()
    setup_runtime_funcs(libcudart)

    try:
        cst.enable(["cudaStreamCreate", "cudaStreamSynchronize", "cudaStreamDestroy"],
                   domains=("runtime",), site="enter", only_current_thread=True)
    except RuntimeError as e:
        pytest.skip(f"CUPTI not available/working here: {e}")

    def main_stream_marker():
        do_stream_ops(libcudart)

    def worker_stream_marker():
        do_stream_ops(libcudart)

    # Main thread: expect logs with marker name
    main_stream_marker()
    out1 = capfd.readouterr().err
    assert "[cuda_stacktrace]" in out1 and "cudaStreamCreate" in out1 and "main_stream_marker" in out1

    # Worker thread: expect no logs due to filter
    capfd.readouterr()  # clear
    t = threading.Thread(target=worker_stream_marker, name="cst-worker-3")
    t.start()
    t.join()
    out2 = capfd.readouterr().err
    cst.disable()
    assert "cudaStreamCreate" not in out2 and "cudaStreamSynchronize" not in out2
