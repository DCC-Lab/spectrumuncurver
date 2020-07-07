__author__ = "Marc-André Vigneault"
__copyright__ = "Copyright 2020, Marc-André Vigneault", "DCCLAB", "CERVO"
__credits__ = ["Marc-André Vigneault"]
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Marc-André Vigneault"
__email__ = "marc-andre.vigneault.02@hotmail.com"
__status__ = "Production"

import logging
import logging.config
from logging.handlers import RotatingFileHandler
import os
import sys

log = logging.getLogger(__name__)


def init_logging(level):
    logger = logging.getLogger()
    logger.setLevel(logging.NOTSET)

    # create console handler
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s\t\t (%(name)-15.15s) (thread:%(thread)d) (line:%(lineno)5d)\t\t[%(levelname)-5.5s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # create debug file handler in working directory
    logFolderPath = "."+os.sep+"log"
    if not os.path.exists(logFolderPath):
        os.makedirs(logFolderPath)
    handler = RotatingFileHandler(logFolderPath + os.sep + "{0}.log",
                                  maxBytes=10000, backupCount=5)
    handler.setLevel(logging.ERROR)
    formatter = logging.Formatter(
        "%(asctime)s\t\t (%(name)-25.25s) (thread:%(thread)d) (line:%(lineno)5d)\t\t[%(levelname)-5.5s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    log.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


init_logging()
sys.excepthook = handle_exception