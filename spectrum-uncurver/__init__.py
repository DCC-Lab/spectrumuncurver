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
from .spectrumUncurver import SpectrumUncurver

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


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


init_logging()

