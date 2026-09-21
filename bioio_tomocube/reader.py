#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Format-dispatching entry point for the bioio-tomocube plugin.

Tomocube data arrives in two shapes:

* ``.TCF`` — the native HDF5 container written by TomoStudio, read by
  :class:`bioio_tomocube.tcf_reader.TCFReader`.
* ``.TIFF`` — per-modality ImageJ-style TIFF exports written by the HTX
  ``ProcessingServer``, read by :class:`bioio_tomocube.tiff_reader.TiffReader`.

:class:`Reader` is the class registered with bioio.  Instantiating it returns
the concrete backend for the given path, so ``Reader(path)`` always yields an
object with the same scene naming and API regardless of the on-disk format.
"""

import logging
from typing import Any, Optional, Type

from bioio_base import exceptions, io, reader, types
from fsspec.spec import AbstractFileSystem

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

_TCF_EXTENSIONS = (".tcf",)
_TIFF_EXTENSIONS = (".tiff", ".tif")


def _select_backend(path: str) -> Type["Reader"]:
    """Pick the concrete reader class for *path* based on its extension."""
    lowered = path.lower()
    if lowered.endswith(_TCF_EXTENSIONS):
        from bioio_tomocube.tcf_reader import TCFReader

        return TCFReader
    if lowered.endswith(_TIFF_EXTENSIONS):
        from bioio_tomocube.tiff_reader import TiffReader

        return TiffReader
    raise exceptions.UnsupportedFileFormatError(
        "bioio-tomocube",
        path,
        "File does not have a .TCF or .TIFF/.TIF extension.",
    )


class Reader(reader.Reader):
    """Read Tomocube holotomography data (``.TCF`` or HTX ``.TIFF`` exports).

    Parameters
    ----------
    image : Path or str
        Path to a ``.TCF`` file or to any TIFF of an HTX export.  Any
        fsspec-compatible URI is accepted.
    fs_kwargs : Dict[str, Any]
        Keyword arguments forwarded to the fsspec filesystem.  Default: ``{}``.
    **kwargs
        Backend-specific options, e.g. ``timelapse=True`` or
        ``refractive_index=True`` for TIFF exports (see
        :class:`~bioio_tomocube.tiff_reader.TiffReader`).

    Raises
    ------
    exceptions.UnsupportedFileFormatError
        If the file is neither a ``.TCF`` file nor a Tomocube TIFF export.

    Notes
    -----
    ``Reader(path)`` returns an instance of the matching backend
    (:class:`~bioio_tomocube.tcf_reader.TCFReader` or
    :class:`~bioio_tomocube.tiff_reader.TiffReader`); both are subclasses of
    this class, so ``isinstance(rdr, Reader)`` holds.

    Both backends expose imaging modalities as **scenes** with shared names:

    * ``"3D"`` — 3-D refractive-index volume (``TZYX``)
    * ``"2DMIP"`` — 2-D refractive-index max projection (``TYX``)
    * ``"3DFL/CH0"``, ``"3DFL/CH1"``, … — 3-D fluorescence channels (``TZYX``)
    * ``"2DFLMIP"`` / ``"2DFLMIP/CHn"`` — 2-D fluorescence max projections
    """

    def __new__(
        cls, image: Optional[types.PathLike] = None, *args: Any, **kwargs: Any
    ) -> "Reader":
        # ``image`` is optional only so that unpickling (which calls
        # ``cls.__new__(cls)`` with no arguments) works for the backends.
        if cls is Reader:
            if image is None:
                raise TypeError("Reader() missing required argument: 'image'")
            fs_kwargs = kwargs.get("fs_kwargs") or {}
            _, path = io.pathlike_to_fs(
                image, enforce_exists=True, fs_kwargs=fs_kwargs
            )
            backend = _select_backend(path)
            return super().__new__(backend)
        return super().__new__(cls)

    @staticmethod
    def _is_supported_image(fs: AbstractFileSystem, path: str, **kwargs: Any) -> bool:
        backend = _select_backend(path)
        return backend._is_supported_image(fs, path, **kwargs)
