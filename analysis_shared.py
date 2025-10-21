#!/usr/bin/env python3
"""
Shared constants and helpers used across the unified drive cycle analysis
modules.
"""

from __future__ import annotations

import math
from typing import Any

# Single source of truth for the chaos threshold considered "predictable".
CHAOS_PREDICTABLE: float = 0.7


def fmt_p(value: Any) -> str:
    """
    Format p-values consistently across reports and visualizations.

    Values below 1e-3 are expressed in scientific notation to preserve
    magnitude information; larger values use fixed-point formatting.
    """
    try:
        p_val = float(value)
    except (TypeError, ValueError):
        return "n/a"

    if math.isnan(p_val):
        return "n/a"

    if p_val < 0:
        p_val = 0.0

    return f"{p_val:.1e}" if p_val < 1e-3 else f"{p_val:.4f}"


__all__ = ["CHAOS_PREDICTABLE", "fmt_p"]
