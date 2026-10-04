"""Floor-aligned LP partial / top-off pick decomposition."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PickPolicy:
    """Site pick rules for floor vs sim alignment."""

    partial_pick_model: bool = False
    min_layers_to_layer_pick: int = 3
    max_layers_hand_pick: int = 2
    late_shift_hour: int = 18
    late_shift_small_pick_topoff_max_layers: int = 6
    max_lp_pick_height_in: float | None = None  # None → active_height_limits()


def decompose_lp_pick_fulfillment(
    *,
    face_layers: int,
    waved_layers: int,
    as_of_hour: int,
    policy: PickPolicy,
) -> tuple[int, int, int]:
    """
    Split waved demand into LP-ready, partial-LP miss, and top-off hand pick.

    Returns ``(lp_ready_layers, lp_partial_miss_layers, top_off_hand_pick_layers)``.
    """
    if waved_layers <= 0 or face_layers <= 0:
        return 0, 0, 0

    min_lp = policy.min_layers_to_layer_pick
    if face_layers < min_lp:
        return 0, 0, waved_layers

    if (
        as_of_hour >= policy.late_shift_hour
        and waved_layers <= policy.late_shift_small_pick_topoff_max_layers
    ):
        return 0, 0, waved_layers

    lp_pe = min(1.0, face_layers / waved_layers)
    if lp_pe >= 0.999:
        return waved_layers, 0, 0

    partial = max(1, round(waved_layers * (1.0 - lp_pe)))
    partial = min(partial, waved_layers)
    lp_ready = waved_layers - partial
    return lp_ready, partial, 0
