from setuptools import setup, Extension, find_packages
import os, platform


def dedup_existing(paths):
    out = []
    for path in paths:
        if path and os.path.isdir(path) and path not in out:
            out.append(path)
    return out


CUDA_HOME = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"
CUPTI_HOME = os.environ.get("CUPTI_HOME") or os.path.join(CUDA_HOME, "extras", "CUPTI")
ARCH = platform.machine()

# Common include locations across recent CUDA layouts
include_dirs = dedup_existing(
    [
        os.path.join(CUPTI_HOME, "include"),
        os.path.join(CUDA_HOME, "include"),
        os.path.join(CUDA_HOME, "extras", "CUPTI", "include"),
        os.path.join(CUDA_HOME, "targets", f"{ARCH}-linux", "include"),
        os.path.join(CUDA_HOME, "targets", "x86_64-linux", "include"),
        os.path.join(CUDA_HOME, "targets", "sbsa-linux", "include"),
    ]
)

libraries = ["cupti"]

system = platform.system()
if system == "Windows":
    library_dirs = dedup_existing([os.path.join(CUPTI_HOME, "lib64")])
    runtime_library_dirs = []
elif system == "Darwin":
    # CUDA on macOS is generally unavailable; keep for completeness.
    library_dirs = dedup_existing([os.path.join(CUPTI_HOME, "lib")])
    runtime_library_dirs = []
else:
    # Linux: support both legacy extras/CUPTI paths and newer targets/ layouts
    library_dirs = dedup_existing(
        [
            os.environ.get("CUPTI_LIBRARY_PATH"),
            os.path.join(CUPTI_HOME, "lib64"),
            os.path.join(CUPTI_HOME, "lib"),
            os.path.join(CUDA_HOME, "lib64"),
            os.path.join(CUDA_HOME, "targets", f"{ARCH}-linux", "lib"),
            os.path.join(CUDA_HOME, "targets", "x86_64-linux", "lib"),
            os.path.join(CUDA_HOME, "targets", "sbsa-linux", "lib"),
            "/usr/lib/x86_64-linux-gnu",
        ]
    )
    runtime_library_dirs = library_dirs

ext_modules = [
    Extension(
        # Build the native extension as a submodule inside the package
        name="cuda_stacktrace._native",
        sources=["src/cuda_stacktrace.cpp"],
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
        runtime_library_dirs=runtime_library_dirs,
        extra_compile_args=["-std=c++17"],
        language="c++",
    ),
]

setup(
    name="cuda-stacktrace",
    version="0.1.0",
    description="Print the Python call stack when selected CUDA APIs are called (via CUPTI).",
    long_description="",
    packages=find_packages("src"),
    package_dir={"": "src"},
    ext_modules=ext_modules,
    zip_safe=False,
)
