import os
from pathlib import Path
from loguru import logger
import sys

# Set up logging
_should_log = os.getenv("LRDMS_DEVELOPMENT", False)
if _should_log:
    # set log level to INFO
    logger.remove()
    logger.add(sys.stderr, level="INFO")
else:
    # set log level to WARNING
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

# Set up project paths
PKG_PATH = Path(os.path.abspath(__file__))
if (PKG_PATH.parent / "_data").exists():
    DATA_PATH = PKG_PATH.parent / "_data"
elif (PKG_PATH.parents[2] / "data").exists():
    DATA_PATH = PKG_PATH.parents[2] / "data"
else:
    logger.warning("`lrdms` data path not found. Will not be able to auto-load data from `lrdms` package.")
    DATA_PATH = None
