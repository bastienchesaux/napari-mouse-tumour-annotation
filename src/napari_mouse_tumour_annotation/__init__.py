try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"


import warnings

from ._sample_data import make_sample_data
from ._widget import MouseTumourAnnotationQWidget

__all__ = (
    "make_sample_data",
    "MouseTumourAnnotationQWidget",
)

warnings.filterwarnings(
    "ignore", category=FutureWarning, message=".*cuda.cudart.*"
)
