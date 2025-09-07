# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

__version__ = "8.3.117"

import os

# Set ENV variables (place before imports)
if not os.environ.get("OMP_NUM_THREADS"):
    os.environ["OMP_NUM_THREADS"] = "1"  # default for reduced CPU utilization during training

from ultralytics_custom.models import YOLO, YOLOWorld, YOLOE
from ultralytics_custom.utils import ASSETS, SETTINGS
from ultralytics_custom.utils.checks import check_yolo as checks
from ultralytics_custom.utils.downloads import download

settings = SETTINGS
__all__ = (
    "__version__",
    "ASSETS",
    "YOLO",
    "YOLOWorld",
    "YOLOE",
    "checks",
    "download",
    "settings",
)
