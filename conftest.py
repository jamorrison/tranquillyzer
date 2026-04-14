"""Root conftest – preload conda env's libstdc++ before any native extensions
(llvmlite, scipy HiGHS, etc.) are loaded via ctypes/dlopen.

Systems where /lib64/libstdc++.so.6 lacks GLIBCXX_3.4.30 will otherwise
fail at import time for numba, sklearn, and similar packages.
"""

import ctypes
import os

_conda_prefix = os.environ.get("CONDA_PREFIX", "")
if _conda_prefix:
    _libstdcpp = os.path.join(_conda_prefix, "lib", "libstdc++.so.6")
    if os.path.isfile(_libstdcpp):
        ctypes.CDLL(_libstdcpp, mode=ctypes.RTLD_GLOBAL)

    # Also propagate to subprocesses
    _conda_lib = os.path.join(_conda_prefix, "lib")
    _ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if _conda_lib not in _ld_path:
        os.environ["LD_LIBRARY_PATH"] = _conda_lib + (":" + _ld_path if _ld_path else "")
