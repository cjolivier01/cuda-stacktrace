from setuptools import setup, Extension
import os, platform

CUDA_HOME = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"
CUPTI_HOME = os.environ.get("CUPTI_HOME") or os.path.join(CUDA_HOME, "extras", "CUPTI")

# Try to use CUPTI include dir if it exists, otherwise rely on system includes (e.g. /usr/include)
include_dirs = []
cupti_inc = os.path.join(CUPTI_HOME, "include")
if os.path.isdir(cupti_inc):
    include_dirs.append(cupti_inc)

libraries = ["cupti"]
library_dirs = []  # rely on system linker paths by default

system = platform.system()
if system == "Windows":
    library_dirs.append(os.path.join(CUPTI_HOME, "lib64"))
elif system == "Darwin":
    # CUDA on macOS is generally unavailable; keep for completeness.
    library_dirs.append(os.path.join(CUPTI_HOME, "lib"))
else:
    # Linux
    # Try lib64 first, fall back to lib
    lib64 = os.path.join(CUPTI_HOME, "lib64")
    lib   = os.path.join(CUPTI_HOME, "lib")
    if os.path.isdir(lib64):
        library_dirs.append(lib64)
    elif os.path.isdir(lib):
        library_dirs.append(lib)

ext_modules = [
    Extension(
        name="cuda_stacktrace",
        sources=["src/cuda_stacktrace.cpp"],
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
        extra_compile_args=["-std=c++17"],
        language="c++",
    ),
]

setup(
    name="cuda-stacktrace",
    version="0.1.0",
    description="Print the Python call stack when selected CUDA APIs are called (via CUPTI).",
    long_description="",
    ext_modules=ext_modules,
    zip_safe=False,
)
