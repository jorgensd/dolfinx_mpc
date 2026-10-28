# Copyright (C) 2026 Jørgen Schartum Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT
"""Logging for DOLFINx-MPC.

Levels are :class:`dolfinx.log.LogLevel` values. DOLFINx-MPC has its own
logger, so :func:`dolfinx.log.set_log_level` does not change it.
"""

from dolfinx_mpc.cpp.log import get_log_level, set_log_level

__all__ = ["get_log_level", "set_log_level"]
