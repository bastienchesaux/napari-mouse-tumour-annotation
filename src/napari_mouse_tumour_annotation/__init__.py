try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

import os
import platform
import warnings

from ._sample_data import make_sample_data

if platform.system() == "Windows":
    import ctypes
    from importlib.util import find_spec

    try:
        if (
            (spec := find_spec("torch"))
            and spec.origin
            and os.path.exists(
                dll_path := os.path.join(
                    os.path.dirname(spec.origin), "lib", "c10.dll"
                )
            )
        ):
            ctypes.CDLL(os.path.normpath(dll_path))
    except Exception as e:
        print(f"Failed to pre-import c10.dll on Windows target with: {e}")

from ._widget import MouseTumourAnnotationQWidget

__all__ = (
    "make_sample_data",
    "MouseTumourAnnotationQWidget",
)

warnings.filterwarnings(
    "ignore", category=FutureWarning, message=".*cuda.cudart.*"
)
