# -*- coding: utf-8 -*-

"""Top-level package for bioio_tomocube."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("bioio-tomocube")
except PackageNotFoundError:
    __version__ = "uninstalled"

__author__ = "bioio-devs"
__email__ = "brian.whitney@alleninstitute.org"

from .reader import Reader
from .reader_metadata import ReaderMetadata
from .tcf_reader import TCFReader
from .tiff_reader import TiffReader

__all__ = ["Reader", "ReaderMetadata", "TCFReader", "TiffReader"]
