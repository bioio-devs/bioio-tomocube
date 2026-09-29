# -*- coding: utf-8 -*-

"""Top-level package for bioio_tomocube."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("bioio-tomocube")
except PackageNotFoundError:
    __version__ = "uninstalled"

__author__ = "bioio-devs"
