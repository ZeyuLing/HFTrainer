# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""Compatibility namespace for older Lightning installations.

Some debug containers ship ``pytorch_lightning`` and ``lightning_fabric`` but
not the newer unified ``lightning`` package. ProtoMotions imports
``lightning.fabric`` and ``lightning.pytorch``; this local namespace forwards
those imports when the unified package is unavailable.
"""

