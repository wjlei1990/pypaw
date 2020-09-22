#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:copyright:
    Wenjie Lei (lei@princeton.edu), 2016
:license:
    GNU Lesser General Public License, version 3 (LGPLv3)
    (http://www.gnu.org/licenses/lgpl-3.0.en.html)
"""
from __future__ import (absolute_import, division, print_function)
import logging
import collections

__version__ = "0.0.1"

#from .process import ProcASDF       # NOQA
#from .process_serial import ProcASDFSerial       # NOQA
#from .window import WindowASDF      # NOQA
#from .adjoint import AdjointASDF    # NOQA
#from .adjoint_serial import AdjointASDFSerial    # NOQA
#from .measure_adjoint import MeasureAdjointASDF       # NOQA
#from .convert import ConvertASDF, convert_from_asdf   # NOQA
#from .convert import convert_adjsrcs_from_asdf        # NOQA


# Setup the logger.
logger = logging.getLogger("pypaw")
# logger.setLevel(logging.WARNING)
logger.setLevel(logging.INFO)
# Prevent propagating to higher loggers.
logger.propagate = 0
# Console log handler.
ch = logging.StreamHandler()
# Add formatter
FORMAT = "[%(asctime)s] - %(name)s - %(levelname)s: %(message)s"
formatter = logging.Formatter(FORMAT)
ch.setFormatter(formatter)
logger.addHandler(ch)


# mpi class
mpi_ns = collections.namedtuple(
    "mpi_ns", ["comm", "rank", "size", "MPI", "processor"])
