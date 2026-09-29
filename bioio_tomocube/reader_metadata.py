#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List

import bioio_base.reader_metadata

###############################################################################


class ReaderMetadata(bioio_base.reader_metadata.ReaderMetadata):
    """Plugin-level metadata: supported extensions and the reader class."""

    @staticmethod
    def get_supported_extensions() -> List[str]:
        return [".TCF", ".tcf"]

    @staticmethod
    def get_reader() -> bioio_base.reader.Reader:
        from .reader import Reader

        return Reader
