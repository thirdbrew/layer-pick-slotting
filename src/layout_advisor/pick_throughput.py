"""Pick-rate parameters for the simulator and layout sizing.

EVERY NUMBER IN THIS MODULE IS AN ILLUSTRATIVE PLACEHOLDER.
-----------------------------------------------------------
None of these values was measured at any facility. They are round, plausible
figures chosen so the simulator produces sensible output on generated data, and
they are the first thing you should replace when pointing this engine at a real
site. Treating them as calibration would be a mistake -- an unmeasured rate that
looks precise is worse than one that looks obviously provisional, so these are
deliberately round.

Dollar and labour-hour translations of these rates are deliberately absent:
re-basing them on invented numbers would only make the placeholders look precise.

OVERRIDING THEM
---------------
Every constant reads an environment variable first, so a study can be re-run at
a different rate without editing source:

    LP_PICK_CASES_PER_HOUR=1500 python -m slotting.cli simulate ...

Anything derived (`PICK_LAYERS_PER_HOUR`) is recomputed from the overridden
inputs at import, never hand-scaled -- scale one and they silently disagree.
"""

from __future__ import annotations

import os


def _env_float(name: str, default: float) -> float:
    """Read a float override, falling back to the placeholder default."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:                                   # pragma: no cover
        raise ValueError(
            f"{name} must be a number, got {raw!r}"
        ) from exc


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:                                   # pragma: no cover
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


# --------------------------------------------------------------------------
# Mechanised (clamp) layer picking, cases per hour. PLACEHOLDER.
# A clamp lifts whole layers, so its case rate is a large multiple of a hand
# rate; 2,000 is a round stand-in for "roughly an order of magnitude faster".
# --------------------------------------------------------------------------
PICK_CASES_PER_HOUR = _env_float("LP_PICK_CASES_PER_HOUR", 2000.0)

# Conventional hand case-picking, cases per hour. PLACEHOLDER.
# The baseline the mechanised rate is compared against.
HAND_PICK_CASES_PER_HOUR = _env_float("LP_HAND_PICK_CASES_PER_HOUR", 120.0)

# Cases per layer, used only when a SKU master is not supplied. PLACEHOLDER.
# Real books vary widely by product; sizing from this default rather than from
# a weighted SKU master is a known approximation, not a calibration.
PLANNING_CASES_PER_LAYER_DEFAULT = _env_float("LP_CASES_PER_LAYER", 10.0)

# Crew and geometry defaults for layout sizing. PLACEHOLDERS.
DEFAULT_PICKER_COUNT = _env_int("LP_PICKER_COUNT", 1)
DEFAULT_CENTER_GAP_FT = _env_float("LP_CENTER_GAP_FT", 0.0)


def pick_layers_per_hour(
    cases_per_hour: float = PICK_CASES_PER_HOUR,
    cases_per_layer: float = PLANNING_CASES_PER_LAYER_DEFAULT,
) -> float:
    """Layers per hour implied by a case rate and a case-per-layer ratio.

    Derived, never independently configured: if the case rate moves, this moves
    with it. Hand-setting a layer rate alongside a case rate is how the two
    quietly stop describing the same operation.
    """
    if cases_per_layer <= 0:
        raise ValueError("cases_per_layer must be positive")
    return cases_per_hour / cases_per_layer


PICK_LAYERS_PER_HOUR = pick_layers_per_hour()


def hand_pick_multiple() -> float:
    """How many times faster the mechanised rate is, per picker.

    Exposed as a function rather than a constant so it cannot drift away from
    the two rates it is computed from.
    """
    if HAND_PICK_CASES_PER_HOUR <= 0:
        raise ValueError("HAND_PICK_CASES_PER_HOUR must be positive")
    return PICK_CASES_PER_HOUR / HAND_PICK_CASES_PER_HOUR
