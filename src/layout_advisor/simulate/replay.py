"""Replay simulation of the layer-pick floor over a pick day or a month."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from slotting.backtest import (
    BacktestMode,
    BacktestReport,
    HourlyStagingBoost,
    IdleAssistConfig,
    ShiftPullConfig,
    backtest_historical,
    orders_for_as_of,
)
from slotting.models import OrderArea, OrderLine, Pallet, Slot, SlotType, WorldState
from slotting.rules import (
    BACKUP_ROW_AISLE_RULE_CONFIG,
    BACKUP_ROW_RULE_CONFIG,
    DEFAULT_RULE_CONFIG,
    RESUPPLY_RACK_AISLE_RULE_CONFIG,
    RESUPPLY_RACK_RULE_CONFIG,
    RuleConfig,
)
from layout_advisor.layout.slot_geometry import build_aisle_slot_map, source_face_geography
from slotting.generate_slots import build_slots
from slotting.rack_depth import RankedSku, assign_rack_slots_colocated
from slotting.resupply_rack import aisle_stations_per_flank, build_rack_slots_aligned
from slotting.simulation import SimulationResult
from slotting.sweep_floor import build_sweep_floor
from slotting.visibility_planning import (
    final_shift_as_of,
    rank_for_final_shift_setup,
    rank_setup_skus,
    rank_ship_day_setup_skus,
    visibility_snapshot,
)
from slotting.wave_release import bucket_orders_by_ship_day, order_ship_date, spread_orders_to_hourly_waves

from slotting.pallet_geometry import (
    MAX_LP_PICK_HEIGHT_IN,
    MAX_ORDER_PALLET_HEIGHT_IN,
    PalletHeightLimits,
    pallet_height_limits,
)

from layout_advisor.pick_throughput import (
    DEFAULT_CENTER_GAP_FT,
    DEFAULT_PICKER_COUNT,
    PICK_CASES_PER_HOUR,
    PICK_LAYERS_PER_HOUR,
    PLANNING_CASES_PER_LAYER_DEFAULT,
)

if TYPE_CHECKING:
    from layout_advisor.planning.dynamic_slotting import SetupSlottingPlan


@dataclass(frozen=True)
class PickDayReplay:
    """Full replay artifacts — shared by sim CSV, floor visual, and daily plans."""

    report: BacktestReport
    initial: WorldState
    setup_at: datetime
    pad_swaps: tuple
    wave_donor_opens: tuple
    assignments: tuple
    preopen: tuple[tuple[str, int, str], ...]
    visible_skus: int
    visible_layers: int
    # The day's book: layers ordered on this ship day (THIS_LINE). Denominator
    # for fill rate — without it the artifacts can report what was picked but not
    # whether that was the whole day.
    ordered_layers: int = 0
    donor_pad_events: tuple = ()
    preferred_faces: dict[str, str] = field(default_factory=dict)
    face_station_by_id: dict[str, int] = field(default_factory=dict)
    face_flank_by_id: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ReplayConfig:
    # Defaults match the 70:30 module of REFERENCE_REPLAY_CONFIG. Pass 75/25 explicitly if needed.
    source_slots: int = 70
    donor_slots: int = 30
    waves_per_day: int = 12
    send_start_hour: int = 6
    due_offset_hours: float = 4.0
    setup_source_slots: int = 70
    reserve_default_layers: int = 8
    final_shift_hour: int = 22
    backup_row_enabled: bool = False
    reserve_pallets_per_sku: int = 1
    stage_single_order_backups: bool = True
    max_backups_staged: int = 0  # 0 = fill all available; else cap resupply pre-stage count
    allow_aisle_staging: bool = False  # face may sit in aisle briefly during backup swap
    resupply_rack_enabled: bool = False
    resupply_rack_capacity: int = 0  # 0 = auto from reserve cap + peak visible SKUs
    reserve_cap_ratio: float = 1.0
    rack_depth_tiers: int = 1
    rack_only_reserve: bool = True  # all reserve in side racks — no face-adjacent rows
    synthetic_wave_spread: bool = False  # True = balanced hourly spread; False = use due_at from feed
    preopen_donor_enabled: bool = True  # open first-wave donor builds at setup (matches planning)
    early_wave_hours: int = 8  # setup rank: prioritize sends before pick_start + N hours
    preopen_wave_count: int = 3  # donor builds to pre-open for first N native waves
    wave_clustering: bool = True  # cluster same-wave SKUs on contiguous aisle stations
    order_affinity_clustering: bool = True  # complements adjacent; repeat SKUs → prime faces
    rank_anchored_chain_seeding: bool = False  # seed affinity chain by setup rank
    rank_anchored_chain_spike_only: bool = True  # when seeding on, gate to spike ship days
    setup_ship_day_foresight: bool = True  # full ship-day book at 22:00 setup
    setup_arrival_censor: bool = False  # censor setup by order arrived_at
    build_anchored_slotting: bool = False  # median build + Hungarian per wave (travel objective)
    # Light-day-safe BA: stickiness toward stay-put seed on light /
    # pre-light days — keeps BA on whole-chain (no mode toggle). 0 threshold = off.
    build_anchored_light_day_sku_threshold: int = 0
    build_anchored_light_day_stickiness: float = 4.0
    build_anchored_prelight_stickiness: bool = False  # also sticky when next ship day is light
    order_anchored_window_assign: bool = False  # Hungarian within wave window, anchors from placed order-mates
    setup_stay_put_faces: bool = False  # repeat SKUs keep yesterday's face at setup (M-SLAP reassignment cost)
    velocity_twin_faces_top_k: int = 0  # second concurrent face for top-k layer-velocity SKUs (Bartholdi-Hackman)
    velocity_twin_face_reclaim: bool = False  # mid-day REMOVE twin when slot needed in horizon
    velocity_twin_reclaim_horizon_hours: int = 4  # look-ahead for twin reclaim collisions
    # Scaled twin-k: per-day k = min(top_k, staged - buffer); 0 = off (raw k every day).
    # A buffer near the light-day staged count zeroes twins on light days, where they cost faces.
    velocity_twin_scaled_k_buffer: int = 0
    # Rack promote may evict the coldest waved face when all faces are waved-locked
    # (near-term layers strictly below the incoming SKU's; landing slot required)
    waved_displacement: bool = False
    # DISP-RACK-SWAP arm: displaced pallet may land in the promoting pallet's
    # vacated rack slot when no other landing exists (default off).
    waved_displacement_rack_swap: bool = False
    # Soft mid-day preferred refresh from upcoming waved window (virtual faces only)
    wave_window_preferred_refresh: bool = False
    wave_window_preferred_refresh_hours: tuple[int, ...] = (11, 14)
    # Order-ahead schedule (parallel to waves — off by default)
    order_schedule_enabled: bool = False
    sim_step_minutes: int = 60  # 10 when order_schedule_enabled
    order_lookahead_hours: float = 24.0
    # Defer an order's donor-pad open until every line has pickable supply;
    # short orders hold no pad, buildable orders take it (an operating rule)
    order_supply_complete_gate: bool = False
    # Inside N hours of build due the gate stops deferring (waiting guarantees
    # a miss); 0 = never override
    order_gate_due_override_hours: float = 0.0
    max_build_weight_lb: float = 2500.0
    max_build_height_in: float = MAX_ORDER_PALLET_HEIGHT_IN
    max_lp_pick_height_in: float = MAX_LP_PICK_HEIGHT_IN
    skus_path: str | None = None
    pad_free_on_build_complete: bool = True
    donor_open_retry_enabled: bool = True  # retry wave opens when assigned pad still occupied
    order_whole_order_shortage_wait: bool = True
    order_shortage_policy: str = "whole_order"
    order_shortage_late_block_hours: float = 2.0
    dynamic_pad_roles: bool = True  # time-varying source:donor on fixed 100 pads
    hourly_demand_pad_schedule: bool = True  # ratio follows hourly waved SKU/layer demand
    demand_triggered_pad_schedule: bool = True  # legacy single restore when hourly off
    restore_waved_sku_threshold: int = 65  # waved SKUs triggering pad restore (legacy mode)
    prepeak_role_change_hour: int = 14  # fallback hour if threshold never reached (legacy mode)
    pad_schedule_early_source_delta: int = 5  # max pads to swing donor ↔ source per phase
    pad_schedule_spike_early_source_delta: int = 10  # wider swing on heavy spike days
    pad_schedule_spike_layer_threshold: int = 0  # 0 = use rolling_preopen_spike_layer_threshold
    pad_schedule_waved_sku_buffer: int = 3  # transitional source faces = waved SKUs + buffer
    pad_schedule_lull_layer_threshold: int = 110  # below → donor-heavy for ahead prep
    pad_schedule_peak_layer_threshold: int = 175  # above → max source faces
    pad_schedule_peak_waved_sku_threshold: int = 58  # above → max source faces
    pad_schedule_transitional_layer_high: int = 145  # above → max source faces (mid-day ramp)
    pad_schedule_min_phase_hours: int = 2  # min hours between pad role changes (labor cap)
    # Pull a scheduled donor->source restore forward when the pad idles empty in
    # donor role while a waved SKU is seat-starved (an evening admission deadlock).
    pad_role_restore_on_starve: bool = False
    wave_colocated_donor_opens: bool = True  # per-wave donor opens on colocated pads
    wave_multi_pallet_donor_opens: bool = False  # one pad per build-pallet slice (A/B before enable)
    rolling_donor_preopen_enabled: bool = False  # off by default — better afternoon build-face LP in sim
    rolling_preopen_spike_sku_threshold: int = 50  # ship-day peak waved SKUs
    rolling_preopen_spike_layer_threshold: int = 280  # ship-day peak waved layers (corrected orders)
    spike_preopen_wave_count: int = 8  # post-setup waves scanned for setup pre-open on spike days
    spike_ahead_preopen_enabled: bool = True  # prioritize afternoon/heavy waves at setup on spike days
    donor_pad_turnover_enabled: bool = True  # hourly fill of empty donor pads from queue
    donor_pad_turnover_lead_hours: int = 1  # stage queued SKUs waved or sending within N hours
    rolling_preopen_lead_hours: int = 3  # planning: hours before send for afternoon waves
    rolling_preopen_afternoon_hour: int = 12  # afternoon bias + planning roll cutoff
    rolling_preopen_hourly_layer_threshold: int = 176  # min waved layers to run peak pad turnover (~88% of 200 lyr/hr)
    pick_cases_per_hour: float = PICK_CASES_PER_HOUR  # serial clamp-picker cases/hour
    planning_cases_per_layer: float = PLANNING_CASES_PER_LAYER_DEFAULT
    pick_layers_per_hour: float = PICK_LAYERS_PER_HOUR  # cases/hr ÷ planning cpl
    pick_shift_hours: tuple[int, int] | None = None  # picker on floor [start, end); None = always
    repeat_donor_opens_mode: str = "spike"  # off | spike | always — repeat wave opens per SKU
    lull_ahead_prep_enabled: bool = True  # pull donor builds into slack hours before peak waves
    lull_ahead_utilization_pct: float = 0.60  # prep when waved layers < this × hourly pick capacity
    lull_ahead_max_lookahead_hours: int = 10  # prep builds for waves sending within N hours
    lull_ahead_max_builds_per_hour: int = 8  # cap donor opens staged per lull hour
    picker_count: int = DEFAULT_PICKER_COUNT  # 1 = single layer-picker on aisle
    center_gap_ft: float = DEFAULT_CENTER_GAP_FT  # 20.0 = physical gap for dual-picker mode
    track_pallet_inventory: bool = True  # deplete layers_remaining hourly from waved demand
    carry_inventory_across_days: bool = True  # month sim chains floor state day to day
    setup_evict_stale_faces: bool = True
    setup_full_reslot: bool = True
    jit_rack_staging: bool = True
    jit_rack_horizon_hours: int = 12
    setup_rack_stage_all_planned: bool = True  # stage every planned overflow rack at setup (not JIT-only)
    setup_maximize_rack_fill: bool = True  # fill spare rack bays with ranked unslotted reserve
    setup_promote_rack_to_faces: bool = True  # move rack stock onto planned empty faces before new rack staging
    setup_fill_empty_source_faces: bool = True  # after pad swaps, backfill empty source faces from reserve
    low_layer_refresh_enabled: bool = True  # mid-day low-layer face swaps (off in the reference config — wasted travel)
    reserve_max_per_sku: int = 1
    prune_orphan_reserve: bool = True
    midday_realignment_hour: int = 11  # 0 = disabled
    afternoon_realignment_hour: int = 14  # 0 = disabled; pre-peak face reslot
    prioritize_pending_completion: bool = True
    carry_rack_bootstrap: bool = True
    visible_prestage_enabled: bool = False  # off by default: competes for faces with waved demand
    demand_weighted_reserve: bool = True
    demand_weighted_max_pallets_per_sku: int = 3
    spike_day_max_pallets_per_sku: int = 4  # 0 = use max only; applies on spike ship days
    reserve_layer_budget: int = 0  # 0 = off; too low a budget starves waved demand of reserve
    aggressive_zero_demand_reclaim: bool = True
    zero_demand_reclaim_horizon_hours: int = 4
    proactive_zero_demand_remove: bool = False
    proactive_zero_demand_realignment_only: bool = False
    proactive_zero_demand_realignment_hours: tuple[int, ...] = ()
    # Squatter reclaim: evict the nearest zero-demand face when a
    # waved SKU has no face and faces are in deficit, instead of only ever
    # freeing that SKU's own preferred face.
    zero_demand_any_face_reclaim: bool = False
    zero_demand_any_face_max_per_step: int = 4  # 0 = uncapped
    # Skip evictions that cannot produce a layer pick: a SKU owing fewer than
    # min_layers_to_layer_pick is hand-picked regardless of holding a face.
    zero_demand_any_face_min_layers: int = 0
    slot_reclaim_rest_of_day: bool = True
    colocated_donor_repurpose: bool = True
    late_shift_hour: int = 18
    late_shift_retain_inventory: bool = True
    # 0 = any-depth sentinel → resolves to 8 layers (see face_stock_needs_resupply)
    late_shift_resupply_threshold: int = 0
    floor_partial_pick_model_enabled: bool = False
    min_layers_to_layer_pick: int = 3
    max_layers_hand_pick: int = 2
    late_shift_small_pick_topoff_max_layers: int = 6
    late_shift_realignment_hour: int = 17  # 0 = disabled; evening wave face reslot
    evening_realignment_hour: int = 0  # 0 = disabled; pre-evening-peak face reslot (e.g. 19)
    night_realignment_hour: int = 0  # 0 = disabled; final-wave face reslot (e.g. 20)
    prune_carry_stale_unslotted: bool = False  # drop carry orphans outside today's plan
    carry_keep_next_ship_day: bool = False  # retain unslotted carry for tomorrow's ship SKUs
    carry_reserve_next_ship_day: bool = False  # mint reserve for next ship day at carry setup
    carry_next_day_reserve_top_n: int = 40  # cap next-day SKUs added to reserve pool
    carry_reserve_topoff: bool = False  # mint to demand-weighted target when carry is thin
    carry_evening_next_day_boost_enabled: bool = False  # static top-N; use carry_dynamic_evening_prep
    carry_evening_next_day_hour: int = 17
    carry_evening_next_day_boost_top_n: int = 30
    carry_evening_next_day_rack_bootstrap: bool = False
    carry_dynamic_evening_prep_enabled: bool = False
    carry_dynamic_evening_prep_top_n: int = 30
    carry_evening_next_morning_forecast_enabled: bool = False
    carry_evening_next_morning_forecast_top_n: int = 25
    carry_evening_next_morning_overnight_only: bool = False
    # OLS honesty mint (default off — full ship-day foresight stays the default)
    carry_evening_ols_forecast_enabled: bool = False
    carry_evening_ols_early_sku_fraction: float = 0.25
    carry_early_wave_rack_staging: bool = False  # carry setup: rack-stage overnight/early-wave SKUs
    carry_early_wave_horizon_hours: int = 4
    carry_early_wave_rack_top_n: int = 40
    carry_early_wave_overnight_only: bool = False
    rack_promote_near_term_hour: int = 14
    rack_promote_near_term_hours: int = 4
    flank_local_rack_promote_batching: bool = True
    morning_rack_promote_batch_end_hour: int = 11
    wave_lp_same_ship_date_only: bool = False
    wave_lp_report_ship_day_split: bool = True
    # Mid-shift pallet pulls from upstream when waved demand has no floor stock
    shift_pallet_pull_enabled: bool = False
    shift_pallet_pull_on_carry: bool = False  # disable pulls when inventory carries from prior day
    shift_pallet_pull_carry_capped: bool = False  # low-rate pulls on non-prep carry days
    shift_pallet_pull_max_per_hour_carry: int = 2
    shift_pallet_pull_stage_immediately: bool = True  # pull + ADD to face in one trip
    shift_pallet_pull_max_per_sku: int = 2
    shift_pallet_pull_max_per_sku_late: int = 3
    shift_pallet_pull_late_shift_min_layers: int = 6
    late_shift_residual_rescue: bool = False  # +1 pull for small starved residuals after late_shift_hour
    lp_top_pick_model: bool = False  # pick-depth model: top-of-stack bites instead of bottom pick zone
    shift_pallet_pull_max_per_hour_day: int = 4
    shift_pallet_pull_max_per_hour_afternoon: int = 6
    shift_pallet_pull_max_per_hour_evening: int = 10
    # Idle-assist opportunistic staging: when reactive shortage pulls leave unused
    # hourly capacity, bring the *next* pallet for thin active faces onto rack.
    idle_assist_staging_enabled: bool = False
    idle_assist_max_per_hour: int = 2
    idle_assist_face_layer_threshold: int = 4
    idle_assist_prefer_rack: bool = True
    idle_assist_require_unused_pull_budget: bool = True
    idle_assist_min_free_rack_slots: int = 0  # rack headroom guard (0 = off)
    late_shift_staging_boost_enabled: bool = False
    late_shift_staging_hour: int = 17
    carry_sunday_reset_enabled: bool = False  # legacy: flush entire floor at spike setup
    carry_sunday_reset_on_spike_only: bool = True  # only on spike eval days (not every carry day)
    spike_reserve_rebootstrap: bool = False  # drop unplanned carry reserve; mint spike demand pool
    spike_release_slotted_outside_keep: bool = True  # selective slotted release when rebootstraping
    carry_prune_rank_buffer: int = 50  # max setup rank to keep unslotted on normal carry days
    spike_carry_prune_rank_buffer: int = 15  # tighter prune on spike transition
    shift_pallet_pull_after_spike_reset: bool = True  # pulls on spike eval day (carry chain end)
    spike_prep_pull_enabled: bool = False  # Thu/Fri/Sun manifest pulls before spike eval
    spike_prep_pull_weekdays: tuple[int, ...] = (1, 2, 3, 4, 6)  # Tue–Fri + Sun
    spike_prep_pull_max_per_hour: int = 2  # capped pulls on prep days (carry week)
    spike_prep_pull_max_per_hour_heavy: int = 0  # 0 = use max_per_hour for all prep weekdays
    spike_prep_pull_heavy_weekdays: tuple[int, ...] = ()  # subset with heavy cap (e.g. Thu/Fri)
    lp_ineligible_items_path: str | None = None  # the non-LP SKU list → WorldState exclusions
    layer_pick_sku_fraction: float = 1.0  # share of outbound layers LP-eligible

    def __post_init__(self) -> None:
        """Reject structurally impossible configs at construction.

        This dataclass has ~190 fields, ~90 of them booleans, and exactly one
        point in that space is pinned by a test (``REFERENCE_REPLAY_CONFIG``). Before
        this check nothing validated any of it: a nonsensical config ran happily
        and produced a number that looked exactly as quotable as a real one.

        Deliberately **errors only, no warnings**. Every "this knob is inert" or
        "this loses a measurement" case fires on the bare defaults, so warning on
        them would train everyone to ignore the channel. What is here is the set
        of values that cannot describe any floor.
        """
        problems: list[str] = []

        if self.source_slots < 1:
            problems.append(f"source_slots={self.source_slots}: a floor with no source face cannot layer-pick")
        if self.donor_slots < 0:
            problems.append(f"donor_slots={self.donor_slots} is negative")
        if self.setup_source_slots < 0:
            problems.append(f"setup_source_slots={self.setup_source_slots} is negative")
        if self.setup_source_slots > self.source_slots:
            problems.append(
                f"setup_source_slots={self.setup_source_slots} exceeds source_slots="
                f"{self.source_slots}: cannot stage more faces at setup than exist"
            )
        if self.sim_step_minutes < 1:
            problems.append(f"sim_step_minutes={self.sim_step_minutes}: tick length must be at least 1")
        if self.reserve_cap_ratio < 0:
            problems.append(f"reserve_cap_ratio={self.reserve_cap_ratio} is negative")
        if self.rack_depth_tiers < 1:
            problems.append(
                f"rack_depth_tiers={self.rack_depth_tiers}: rack is either absent "
                "(rack_only_reserve=False) or at least one tier deep"
            )
        if self.due_offset_hours < 0:
            problems.append(
                f"due_offset_hours={self.due_offset_hours}: a wave cannot release after it is due"
            )
        if not 0 <= self.final_shift_hour <= 23:
            problems.append(f"final_shift_hour={self.final_shift_hour} is not an hour of the day")
        if not 0 <= self.send_start_hour <= 23:
            problems.append(f"send_start_hour={self.send_start_hour} is not an hour of the day")
        if not 0.0 <= self.layer_pick_sku_fraction <= 1.0:
            problems.append(
                f"layer_pick_sku_fraction={self.layer_pick_sku_fraction} is not a share in [0, 1]"
            )
        # Only gated when it is actually read; otherwise it is an inert knob and
        # rejecting it would fail configs that are behaviourally fine.
        if self.synthetic_wave_spread and self.waves_per_day < 1:
            problems.append(
                f"waves_per_day={self.waves_per_day} with synthetic_wave_spread=True: "
                "a synthetic day needs at least one wave"
            )

        if problems:
            raise ValueError(
                "Invalid ReplayConfig:\n  - " + "\n  - ".join(problems)
            )

    @property
    def dual_picker(self) -> bool:
        return self.picker_count > 1 or self.center_gap_ft > 0

    def effective_picker_count(self) -> int:
        return max(1, self.picker_count)

    def effective_pick_layers_per_hour(self, *, base_rate: float) -> float:
        return base_rate * self.effective_picker_count()

    def picker_layout_kwargs(self) -> dict[str, int | float]:
        """Picker geometry for slotting / wave assign (no rack flags)."""
        return {
            "picker_count": self.picker_count,
            "center_gap_ft": self.center_gap_ft,
        }

    def pad_schedule_kwargs(self) -> dict[str, int | bool]:
        """Pad role schedule parameters for planning and simulation."""
        return {
            "hourly_demand": self.hourly_demand_pad_schedule,
            "demand_triggered": self.demand_triggered_pad_schedule,
            "restore_waved_sku_threshold": self.restore_waved_sku_threshold,
            "early_source_delta": self.pad_schedule_early_source_delta,
            "waved_sku_buffer": self.pad_schedule_waved_sku_buffer,
            "lull_layer_threshold": self.pad_schedule_lull_layer_threshold,
            "peak_layer_threshold": self.pad_schedule_peak_layer_threshold,
            "peak_waved_sku_threshold": self.pad_schedule_peak_waved_sku_threshold,
            "transitional_layer_high": self.pad_schedule_transitional_layer_high,
            "min_phase_hours": self.pad_schedule_min_phase_hours,
        }

    def effective_pad_early_source_delta(
        self,
        orders: list[OrderLine],
        ship_date: date,
    ) -> int:
        """Wider pad swing on spike days above the layer threshold."""
        layer_threshold = (
            self.pad_schedule_spike_layer_threshold
            if self.pad_schedule_spike_layer_threshold > 0
            else self.rolling_preopen_spike_layer_threshold
        )
        if not self._is_spike_ship_day(orders, ship_date):
            return self.pad_schedule_early_source_delta
        from layout_advisor.planning.donor_build_queue import ship_day_peak_waved_demand

        _, peak_layers = ship_day_peak_waved_demand(
            orders,
            ship_date,
            pick_start_hour=self.send_start_hour,
            final_shift_hour=self.final_shift_hour,
        )
        if peak_layers >= layer_threshold:
            return self.pad_schedule_spike_early_source_delta
        return self.pad_schedule_early_source_delta

    def pad_schedule_kwargs_for_day(
        self,
        orders: list[OrderLine],
        ship_date: date,
    ) -> dict[str, int | bool]:
        """Pad schedule kwargs with ship-day–conditional early_source_delta."""
        kw = self.pad_schedule_kwargs()
        kw["early_source_delta"] = self.effective_pad_early_source_delta(orders, ship_date)
        return kw

    def aisle_layout_kwargs(self) -> dict[str, int | float | bool]:
        """Keyword args for ``build_aisle_slot_map`` / ``build_aisle_layout``."""
        return {
            **self.picker_layout_kwargs(),
            "rack_only_reserve": self.rack_only_reserve,
        }

    def effective_rack_capacity(self, visible_sku_count: int) -> int:
        from layout_advisor.layout.reserve_policy import max_side_rack_slots, side_rack_slots_for_peak

        if not self.resupply_rack_enabled:
            return 0
        auto = side_rack_slots_for_peak(
            visible_skus=visible_sku_count,
            source_faces=self.setup_source_slots,
            source_count=self.source_slots,
            donor_count=self.donor_slots,
            ratio=self.reserve_cap_ratio,
            rack_only_reserve=self.rack_only_reserve,
        )
        if self.resupply_rack_capacity > 0:
            ceiling = max_side_rack_slots(
                source_count=self.source_slots,
                donor_count=self.donor_slots,
                ratio=self.reserve_cap_ratio,
                rack_only_reserve=self.rack_only_reserve,
            )
            return min(self.resupply_rack_capacity, auto, ceiling)
        return auto

    @property
    def realignment_hours(self) -> tuple[int, ...]:
        hours: list[int] = []
        for hour in (
            self.midday_realignment_hour,
            self.afternoon_realignment_hour,
            self.late_shift_realignment_hour,
            self.evening_realignment_hour,
            self.night_realignment_hour,
        ):
            if hour:
                hours.append(hour)
        return tuple(hours)

    def proactive_realignment_hours(self) -> tuple[int, ...]:
        if self.proactive_zero_demand_realignment_hours:
            return self.proactive_zero_demand_realignment_hours
        return self.realignment_hours

    def pick_policy(self):
        from slotting.floor_pick_model import PickPolicy

        return PickPolicy(
            partial_pick_model=self.floor_partial_pick_model_enabled,
            min_layers_to_layer_pick=self.min_layers_to_layer_pick,
            max_layers_hand_pick=self.max_layers_hand_pick,
            late_shift_hour=self.late_shift_hour,
            late_shift_small_pick_topoff_max_layers=self.late_shift_small_pick_topoff_max_layers,
            max_lp_pick_height_in=self.max_lp_pick_height_in,
        )

    def height_limits(self) -> PalletHeightLimits:
        return PalletHeightLimits(
            max_order_pallet_height_in=self.max_build_height_in,
            max_lp_pick_height_in=self.max_lp_pick_height_in,
            skus_path=Path(self.skus_path) if self.skus_path else None,
            top_pick_model=self.lp_top_pick_model,
        )

    @property
    def rule_config(self) -> RuleConfig:
        if self.resupply_rack_enabled:
            if self.allow_aisle_staging:
                return RESUPPLY_RACK_AISLE_RULE_CONFIG
            return RESUPPLY_RACK_RULE_CONFIG
        if self.backup_row_enabled and self.allow_aisle_staging:
            return BACKUP_ROW_AISLE_RULE_CONFIG
        if self.backup_row_enabled:
            return BACKUP_ROW_RULE_CONFIG
        return DEFAULT_RULE_CONFIG

    @property
    def effective_reserve_per_sku(self) -> int:
        base = self.reserve_max_per_sku
        if self.resupply_rack_enabled:
            return max(base, 1)
        if self.backup_row_enabled:
            return max(base, 2)
        return base

    def effective_twin_faces_top_k(
        self,
        orders: list[OrderLine],
        ship_date: date | None,
    ) -> int:
        """Twin-face k for *ship_date* — scaled by staged-face count when enabled.

        k = min(top_k, staged - buffer), staged ~= min(ship-day SKUs, setup
        source faces). Zeroes twins on light days (few staged SKUs) so they
        never steal faces demand can't repay; full k on heavy days.
        """
        k = self.velocity_twin_faces_top_k
        if k <= 0 or self.velocity_twin_scaled_k_buffer <= 0 or ship_date is None:
            return k
        from slotting.wave_release import order_ship_date

        ship_skus = len({o.item_id for o in orders if order_ship_date(o) == ship_date})
        staged = min(ship_skus, self.setup_source_slots)
        return max(0, min(k, staged - self.velocity_twin_scaled_k_buffer))

    def build_anchored_kwargs(self) -> dict:
        """Slotting kwargs for BA light-day stickiness."""
        return {
            "build_anchored_slotting": self.build_anchored_slotting,
            "build_anchored_light_day_sku_threshold": (
                self.build_anchored_light_day_sku_threshold
            ),
            "build_anchored_light_day_stickiness": self.build_anchored_light_day_stickiness,
            "build_anchored_prelight_stickiness": self.build_anchored_prelight_stickiness,
        }
    def _is_spike_ship_day(self, orders: list[OrderLine], ship_date: date) -> bool:
        from layout_advisor.planning.donor_build_queue import is_spike_ship_day

        return is_spike_ship_day(
            orders,
            ship_date,
            sku_threshold=self.rolling_preopen_spike_sku_threshold,
            layer_threshold=self.rolling_preopen_spike_layer_threshold,
            pick_start_hour=self.send_start_hour,
            final_shift_hour=self.final_shift_hour,
        )

    def effective_repeat_donor_opens(
        self,
        orders: list[OrderLine],
        ship_date: date,
    ) -> bool:
        """Repeat wave donor opens — spike-gated by default.

        Repeat opens let a SKU waved in several waves get a build pad per wave,
        which helps on stress days but also adds waits. Their effect on ordinary
        carry days is unproven, so the gate stays conservative: spike days only."""
        mode = self.repeat_donor_opens_mode
        if mode == "always":
            return True
        if mode == "spike":
            return self._is_spike_ship_day(orders, ship_date)
        return False

    def effective_rank_anchored_chain(
        self,
        orders: list[OrderLine],
        ship_date: date,
    ) -> bool:
        """Rank-anchored chain seeding — optionally spike-gated."""
        if not self.rank_anchored_chain_seeding:
            return False
        if self.rank_anchored_chain_spike_only:
            return self._is_spike_ship_day(orders, ship_date)
        return True

    def effective_demand_weighted_max_pallets(self, orders: list[OrderLine], ship_date: date) -> int:
        base = self.demand_weighted_max_pallets_per_sku
        if self.spike_day_max_pallets_per_sku > 0 and self._is_spike_ship_day(orders, ship_date):
            return max(base, self.spike_day_max_pallets_per_sku)
        return base

    def carry_prune_max_rank(self, *, is_spike: bool) -> int:
        buffer = self.spike_carry_prune_rank_buffer if is_spike else self.carry_prune_rank_buffer
        return self.setup_source_slots + buffer

    def is_spike_prep_pull_day(self, ship_date: date) -> bool:
        return self.spike_prep_pull_enabled and ship_date.weekday() in self.spike_prep_pull_weekdays

    def spike_prep_pull_max_per_hour_for(self, ship_date: date) -> int:
        """Prep-day shift pull cap; heavy weekdays override when configured."""
        if (
            self.spike_prep_pull_max_per_hour_heavy > 0
            and ship_date.weekday() in self.spike_prep_pull_heavy_weekdays
        ):
            return self.spike_prep_pull_max_per_hour_heavy
        return self.spike_prep_pull_max_per_hour

    def shift_pulls_enabled(
        self,
        *,
        carried_state: WorldState | None,
        ship_date: date,
        orders: list[OrderLine],
    ) -> bool:
        if not self.shift_pallet_pull_enabled:
            return False
        if carried_state is None:
            return True
        if self.shift_pallet_pull_on_carry:
            return True
        if self.is_spike_prep_pull_day(ship_date):
            return True
        if (
            carried_state is not None
            and self.shift_pallet_pull_carry_capped
            and not self._is_spike_ship_day(orders, ship_date)
        ):
            return True
        if (
            self.shift_pallet_pull_after_spike_reset
            and self._is_spike_ship_day(orders, ship_date)
        ):
            return True
        return False

    def demand_weighted_pallets_by_item(
        self,
        orders: list[OrderLine],
        item_ids: list[str],
        ship_date: date,
    ) -> dict[str, int]:
        from slotting.inventory_policy import demand_weighted_pallets_per_sku

        spike = self._is_spike_ship_day(orders, ship_date)
        cap = self.effective_demand_weighted_max_pallets(orders, ship_date)
        return {
            item_id: demand_weighted_pallets_per_sku(
                orders,
                item_id,
                ship_date,
                default_layers=self.reserve_default_layers,
                min_per_sku=self.reserve_max_per_sku,
                max_per_sku=cap,
                is_spike_day=spike,
                late_shift_hour=self.late_shift_hour,
                final_shift_hour=self.final_shift_hour,
            )
            for item_id in item_ids
        }


@dataclass(frozen=True)
class ReplayStepRow:
    as_of: datetime
    visible_skus: int
    waved_skus: int
    waved_layers: int
    wave_lp_baseline_pct: float
    wave_lp_projected_pct: float
    build_face_lp_pct: float
    dead_slot_pct: float
    shortage_count: int
    adds: int
    removes: int
    waits: int
    backup_promotes: int
    rack_promotes: int
    resupply_relocates: int
    donor_stages: int
    storage_returns: int
    active_ratio: str
    floor_layers_remaining: int = 0
    reserve_layers_remaining: int = 0
    pending_completion_pallets: int = 0
    pending_completion_layers: int = 0
    lp_ready_layers: int = 0
    hand_unstaged_layers: int = 0
    hand_shortage_layers: int = 0
    hand_excluded_layers: int = 0
    low_layer_slotted_layers: int = 0
    shift_pulls: int = 0
    wave_lp_ship_day_pct: float = 0.0
    wave_lp_bleed_pct: float = 0.0
    waved_layers_ship_day: int = 0
    waved_layers_bleed: int = 0
    lp_ready_station_span: int = 0
    planned_station_span: int = 0
    flank_balance_pct: float = 100.0
    waved_layers_left: int = 0
    waved_layers_right: int = 0


@dataclass(frozen=True)
class ReplaySummary:
    period_label: str
    step_count: int
    step_hours: float
    peak_waved_skus: int
    peak_waved_layers: int
    avg_wave_lp_projected_pct: float
    min_wave_lp_projected_pct: float
    avg_dead_slot_pct: float
    total_adds: int
    total_removes: int
    total_waits: int
    total_backup_promotes: int
    total_rack_promotes: int
    total_resupply_relocates: int
    total_donor_stages: int
    total_storage_returns: int
    output_csv: Path


def layer_pick_floor_pct(sim_lp_pct: float, *, fraction: float) -> float:
    """Facility-wide LP% FLOOR: sim LP scaled by the LP-eligible share.

    The sim treats **every** SKU as layer-pick eligible. `layer_pick_sku_fraction`
    carries the eligible share, but nothing in the simulation reads it, so the
    figure the simulator reports is "LP% if the whole book were eligible", not
    "LP% of what leaves the building".

    **A floor, not a conversion.** Multiplying assumes LP performance on the
    eligible subset equals LP performance measured across all SKUs. If the
    ineligible share skews toward slow movers that would rarely earn a face
    anyway, the true facility figure lands **above** this. It cannot land below,
    which is what makes it safe to quote.

    Supplying a SKU-level eligibility list retires this: an exclusions feed
    gates the sim directly and this estimate becomes unnecessary.
    """
    if fraction >= 1.0:
        return round(sim_lp_pct, 1)
    return round(sim_lp_pct * fraction, 1)


def load_lp_exclusions(path: str | None) -> list:
    """Load non-layer-pickable SKUs from CSV (item_id column)."""
    if not path:
        return []
    from pathlib import Path

    from slotting.models import Exclusion

    p = Path(path)
    if not p.is_file():
        return []
    out: list[Exclusion] = []
    with p.open(newline="", encoding="utf-8") as fh:
        import csv

        for row in csv.DictReader(fh):
            item_id = (row.get("item_id") or row.get("prtnum") or "").strip()
            if not item_id:
                continue
            reason = (row.get("reason") or "not_lp_eligible").strip()
            out.append(Exclusion(item_id=item_id, reason=reason))
    return out


def _spread_orders(orders: list[OrderLine], cfg: ReplayConfig) -> list[OrderLine]:
    if not cfg.synthetic_wave_spread:
        return orders
    return spread_orders_to_hourly_waves(
        orders,
        waves_per_day=cfg.waves_per_day,
        shift_start_hour=cfg.send_start_hour,
        due_offset_hours=cfg.due_offset_hours,
    )


def _orders_in_date_range(
    orders: list[OrderLine],
    *,
    start: date,
    end: date,
) -> list[OrderLine]:
    return [o for o in orders if start <= o.due_at.date() <= end]


def _unique_skus(orders: list[OrderLine]) -> list[str]:
    return sorted({o.item_id for o in orders if o.area == OrderArea.THIS_LINE})


def _setup_reserve_item_ids(
    orders: list[OrderLine],
    visible_orders: list[OrderLine],
    *,
    ship_date,
    next_ship_day: date | None = None,
    next_day_top_n: int = 0,
    setup_ship_day_foresight: bool = True,
    setup_arrival_censor: bool = False,
) -> list[str]:
    """Reserve pallets for ship-day SKUs (plus optional next-day top ranks)."""
    if ship_date is not None:
        ranked = rank_for_final_shift_setup(
            orders,
            ship_date,
            setup_as_of=final_shift_as_of(ship_date),
            setup_ship_day_foresight=setup_ship_day_foresight,
            setup_arrival_censor=setup_arrival_censor,
        )
        item_ids = {item for item, _, _ in ranked}
        if next_ship_day is not None and next_day_top_n > 0:
            next_ranked = rank_ship_day_setup_skus(
                orders,
                next_ship_day,
                setup_as_of=final_shift_as_of(next_ship_day),
            )
            for item_id, _, _ in next_ranked[:next_day_top_n]:
                item_ids.add(item_id)
        return sorted(item_ids)
    return sorted(_unique_skus(visible_orders))


def _single_order_skus(orders: list[OrderLine], *, ship_date) -> set[str]:
    """SKUs with exactly one order line on *ship_date* (quick-swap backup candidates)."""
    by_item: dict[str, int] = {}
    for order in orders:
        if order.area != OrderArea.THIS_LINE:
            continue
        if order_ship_date(order) != ship_date:
            continue
        by_item[order.item_id] = by_item.get(order.item_id, 0) + 1
    return {item for item, count in by_item.items() if count == 1}


def _reserve_pool(pallets: list[Pallet]) -> dict[str, list[Pallet]]:
    from slotting.inventory_policy import supply_pallet_sort_key

    pool: dict[str, list[Pallet]] = {}
    for pallet in pallets:
        if pallet.current_slot:
            continue
        pool.setdefault(pallet.item_id, []).append(pallet)
    for item_id in pool:
        pool[item_id].sort(key=supply_pallet_sort_key)
    return pool


def _take_reserve(pool: dict[str, list[Pallet]], item_id: str) -> Pallet | None:
    candidates = pool.get(item_id, [])
    while candidates:
        pallet = candidates.pop(0)
        if pallet.current_slot:
            continue
        if not candidates:
            pool.pop(item_id, None)
        return pallet
    return None


def _promote_rack_to_planned_faces_at_setup(
    pallets: list[Pallet],
    slots: list[Slot],
    *,
    ranked_skus: list,
    face_by_item: dict[str, str],
    max_sources: int,
) -> int:
    """Move rack pallets onto planned empty source faces — avoids rack→face double-handle."""
    from slotting.inventory_policy import clear_pending_completion

    slot_by_id = {s.slot_id: s for s in slots}
    promoted = 0
    for sku in ranked_skus[:max_sources]:
        face_id = face_by_item.get(sku.item_id)
        if not face_id:
            continue
        face = slot_by_id.get(face_id)
        if face is None or face.slot_type != SlotType.SOURCE or face.occupied_by_pallet_id:
            continue
        rack_pallet = next(
            (
                p
                for p in pallets
                if p.item_id == sku.item_id
                and p.current_slot
                and (rack := slot_by_id.get(p.current_slot)) is not None
                and rack.slot_type == SlotType.RESUPPLY_RACK
            ),
            None,
        )
        if rack_pallet is None:
            continue
        rack_slot = slot_by_id[rack_pallet.current_slot]
        rack_slot.occupied_by_pallet_id = None
        clear_pending_completion(rack_pallet)
        rack_pallet.current_slot = face_id
        face.occupied_by_pallet_id = rack_pallet.pallet_id
        promoted += 1
    return promoted


def _evict_stale_faces_at_setup(
    pallets: list[Pallet],
    slots: list[Slot],
    *,
    keep_item_ids: set[str],
    resupply_rack_enabled: bool,
    as_of: datetime,
    setup_full_reslot: bool = False,
    face_by_item: dict[str, str] | None = None,
    twin_face_by_item: dict[str, str] | None = None,
) -> None:
    """Move carried face pallets not in today's slotting plan off the active face.

    Always clears ghost refs (slot names a missing/depleted pallet). With
    ``setup_full_reslot``, also frees faces that have no planned SKU so
    stay-put carry cannot squat unplanned pads and starve today's staging
    (needed once ``build_reserve_pallets`` keeps slotted carry).
    Twin faces count as planned for their SKU.
    """
    if not keep_item_ids and not setup_full_reslot:
        return
    from slotting.inventory_policy import mark_removed_from_pick

    pallet_by_id = {p.pallet_id: p for p in pallets}
    planned_sku_by_face: dict[str, str] = {}
    if face_by_item:
        for sku, face_id in face_by_item.items():
            planned_sku_by_face[face_id] = sku
    if twin_face_by_item:
        for sku, face_id in twin_face_by_item.items():
            planned_sku_by_face[face_id] = sku

    for slot in slots:
        if slot.slot_type != SlotType.SOURCE or not slot.occupied_by_pallet_id:
            continue
        pallet = pallet_by_id.get(slot.occupied_by_pallet_id)
        if pallet is None or pallet.layers_remaining <= 0:
            slot.occupied_by_pallet_id = None
            if pallet is not None:
                pallet.current_slot = None
            continue
        planned_for_slot = planned_sku_by_face.get(slot.slot_id)
        blocks_planned_slot = (
            planned_for_slot is not None and pallet.item_id != planned_for_slot
        )
        unplanned_squatter = setup_full_reslot and planned_for_slot is None
        if not blocks_planned_slot and not unplanned_squatter:
            continue
        mark_removed_from_pick(pallet, as_of=as_of, source_slot_id=slot.slot_id)
        slot.occupied_by_pallet_id = None
        pallet.current_slot = None


def _apply_setup_shift(
    state: WorldState,
    *,
    max_sources: int,
    backup_row_enabled: bool = False,
    stage_single_order_backups: bool = True,
    ship_date=None,
    max_backups_staged: int = 0,
    resupply_rack_enabled: bool = False,
    resupply_rack_capacity: int = 0,
    rack_only_reserve: bool = True,
    setup_rank_cfg: ReplayConfig | None = None,
    slotting_plan: SetupSlottingPlan | None = None,
    setup_evict_stale_faces: bool = False,
    setup_full_reslot: bool = False,
    jit_rack_staging: bool = False,
    jit_rack_horizon_hours: int = 8,
    carry_rack_bootstrap: bool = True,
    setup_rack_stage_all_planned: bool = True,
    setup_maximize_rack_fill: bool = True,
    setup_promote_rack_to_faces: bool = True,
    carry_setup: bool = False,
) -> WorldState:
    """Pre-slot top visible SKUs onto source faces and racks (final-shift dynamic setup)."""
    from layout_advisor.planning.dynamic_slotting import build_setup_slotting_plan
    from layout_advisor.planning.wave_slotting import (
        build_face_by_item,
        preferred_source_faces,
        primary_wave_release_by_item,
    )

    cfg = setup_rank_cfg or ReplayConfig()
    plan = slotting_plan
    if plan is None and ship_date is not None:
        plan = build_setup_slotting_plan(
            state.orders,
            ship_date,
            source_count=max_sources,
            donor_count=sum(1 for s in state.slots if s.slot_type == SlotType.DESTINATION),
            final_shift_hour=cfg.final_shift_hour,
            early_wave_hours=cfg.early_wave_hours,
            due_offset_hours=cfg.due_offset_hours,
            pick_start_hour=cfg.send_start_hour,
            wave_clustering=cfg.wave_clustering,
            order_affinity=cfg.order_affinity_clustering,
            rank_anchored_chain=cfg.effective_rank_anchored_chain(state.orders, ship_date),
            **cfg.build_anchored_kwargs(),
            order_anchored_window_assign=cfg.order_anchored_window_assign,
            setup_ship_day_foresight=cfg.setup_ship_day_foresight,
            setup_arrival_censor=cfg.setup_arrival_censor,
            velocity_twin_faces_top_k=cfg.effective_twin_faces_top_k(state.orders, ship_date),
            prev_face_by_item=(
                _carried_face_by_item(state) if cfg.setup_stay_put_faces else None
            ),
            **cfg.picker_layout_kwargs(),
            skus_path=Path(cfg.skus_path) if cfg.skus_path else None,
        )

    twin_faces: dict[str, str] = {}
    if plan is not None:
        ranked_skus = list(plan.ranked_skus)
        face_by_item = plan.face_by_item
        rack_by_item = plan.rack_by_item
        twin_faces = dict(plan.twin_face_by_item)
    else:
        ranked = rank_setup_skus(state.orders, state.as_of)
        ranked_skus = [
            RankedSku(rank=index, item_id=item_id, layers=layers, earliest_due=due)
            for index, (item_id, layers, due) in enumerate(ranked, start=1)
        ]
        donor_pad_count = sum(1 for s in state.slots if s.slot_type == SlotType.DESTINATION)
        layout, aisle_map = build_aisle_slot_map(
            source_count=max_sources,
            donor_count=donor_pad_count,
            **cfg.aisle_layout_kwargs(),
        )
        face_by_item, face_order = build_face_by_item(
            ranked_skus,
            state.orders,
            state.as_of.date(),
            source_count=max_sources,
            donor_count=donor_pad_count,
            aisle_map=aisle_map,
            layout=layout,
            wave_clustering=cfg.wave_clustering,
            order_affinity=cfg.order_affinity_clustering,
            rank_anchored_chain=cfg.effective_rank_anchored_chain(
                state.orders, ship_date or state.as_of.date()
            ),
            **cfg.build_anchored_kwargs(),
            order_anchored_window_assign=cfg.order_anchored_window_assign,
            prev_face_by_item=(
                _carried_face_by_item(state) if cfg.setup_stay_put_faces else None
            ),
            pick_start_hour=cfg.send_start_hour,
            final_shift_hour=cfg.final_shift_hour,
            skus_path=Path(cfg.skus_path) if cfg.skus_path else None,
        )
        rack_by_item = {}
        if resupply_rack_enabled:
            from slotting.resupply_rack import overflow_rack_slot_ids

            primary_wave = (
                primary_wave_release_by_item(
                    state.orders,
                    ship_date,
                    due_offset_hours=cfg.due_offset_hours,
                )
                if cfg.wave_clustering and ship_date is not None
                else None
            )
            preferred = preferred_source_faces(
                ranked_skus,
                face_by_item,
                face_order,
                source_count=max_sources,
                primary_wave=primary_wave,
            )
            rack_virtual_faces = dict(preferred)
            rack_virtual_faces.update(face_by_item)
            rack_by_item = assign_rack_slots_colocated(
                ranked_skus[max_sources:],
                overflow_rack_slot_ids(source_count=max_sources, donor_count=donor_pad_count),
                face_order=face_order,
                aisle_map=aisle_map,
                virtual_face_by_item=rack_virtual_faces,
            )

    pallets = [p.model_copy(deep=True) for p in state.pallets]
    slots = [s.model_copy(deep=True) for s in state.slots]
    slot_by_id = {s.slot_id: s for s in slots}
    if setup_evict_stale_faces or setup_full_reslot:
        keep_ids = {sku.item_id for sku in ranked_skus}
        _evict_stale_faces_at_setup(
            pallets,
            slots,
            keep_item_ids=keep_ids,
            resupply_rack_enabled=resupply_rack_enabled,
            as_of=state.as_of,
            setup_full_reslot=setup_full_reslot,
            face_by_item=face_by_item,
            twin_face_by_item=twin_faces,
        )
    reserve = _reserve_pool(pallets)

    from slotting.inventory_policy import clear_pending_completion

    staged_slots: list[Slot] = []
    for sku in ranked_skus[:max_sources]:
        face_id = face_by_item.get(sku.item_id)
        if not face_id:
            continue
        slot = slot_by_id.get(face_id)
        if slot is None or slot.occupied_by_pallet_id:
            continue
        pallet = _take_reserve(reserve, sku.item_id)
        if pallet is None:
            continue
        clear_pending_completion(pallet)
        pallet.current_slot = slot.slot_id
        slot.occupied_by_pallet_id = pallet.pallet_id
        staged_slots.append(slot)

    for item_id, twin_id in twin_faces.items():
        # Velocity twin: second concurrent face for a hot SKU (B-H allocation).
        slot = slot_by_id.get(twin_id)
        if slot is None or slot.occupied_by_pallet_id:
            continue
        pallet = _take_reserve(reserve, item_id)
        if pallet is None:
            continue
        clear_pending_completion(pallet)
        pallet.current_slot = slot.slot_id
        slot.occupied_by_pallet_id = pallet.pallet_id
        staged_slots.append(slot)

    if setup_promote_rack_to_faces and resupply_rack_enabled:
        _promote_rack_to_planned_faces_at_setup(
            pallets,
            slots,
            ranked_skus=ranked_skus,
            face_by_item=face_by_item,
            max_sources=max_sources,
        )

    if backup_row_enabled:
        if stage_single_order_backups and ship_date is not None:
            on_face = {
                p.item_id
                for p in pallets
                if p.current_slot and any(s.slot_id == p.current_slot for s in staged_slots)
            }
            singles = sorted(_single_order_skus(state.orders, ship_date=ship_date) - on_face)
            backup_targets = [s for s in staged_slots if not s.backup_pallet_id]
            staged_backups = 0
            for item_id in singles:
                if not backup_targets:
                    break
                if max_backups_staged and staged_backups >= max_backups_staged:
                    break
                backup = _take_reserve(reserve, item_id)
                if backup is None:
                    continue
                slot = backup_targets.pop(0)
                slot.backup_pallet_id = backup.pallet_id
                staged_backups += 1

        if not max_backups_staged:
            for slot in staged_slots:
                if slot.backup_pallet_id:
                    continue
                face = next(p for p in pallets if p.pallet_id == slot.occupied_by_pallet_id)
                backup = _take_reserve(reserve, face.item_id)
                if backup is not None:
                    slot.backup_pallet_id = backup.pallet_id

    if resupply_rack_enabled:
        from slotting.inventory_policy import clear_pending_completion, jit_rack_item_ids

        on_floor = {
            p.item_id
            for p in pallets
            if p.current_slot
            or any(s.backup_pallet_id == p.pallet_id for s in slots)
        }
        ranked_ids = [sku.item_id for sku in ranked_skus]
        jit_items = (
            set(ranked_ids[max_sources:])
            if setup_rack_stage_all_planned
            else (
                jit_rack_item_ids(
                    ranked_ids,
                    state,
                    source_face_count=max_sources,
                    jit_horizon_hours=jit_rack_horizon_hours,
                )
                if jit_rack_staging
                else set(ranked_ids[max_sources:])
            )
        )
        for sku in ranked_skus[max_sources:]:
            if jit_rack_staging and sku.item_id not in jit_items:
                continue
            rack_id = rack_by_item.get(sku.item_id)
            if not rack_id:
                continue
            if sku.item_id in on_floor:
                continue
            rack = slot_by_id.get(rack_id)
            if rack is None or rack.occupied_by_pallet_id:
                continue
            pallet = _take_reserve(reserve, sku.item_id)
            if pallet is None:
                continue
            clear_pending_completion(pallet)
            rack.occupied_by_pallet_id = pallet.pallet_id
            pallet.current_slot = rack.slot_id
            on_floor.add(sku.item_id)

        cfg = setup_rank_cfg or ReplayConfig()
        if (
            carry_setup
            and cfg.carry_early_wave_rack_staging
            and ship_date is not None
            and not cfg._is_spike_ship_day(state.orders, ship_date)
        ):
            from slotting.inventory_policy import stage_carry_early_wave_unslotted_to_rack

            stage_carry_early_wave_unslotted_to_rack(
                pallets,
                slots,
                state.orders,
                ship_date,
                as_of=state.as_of,
                horizon_hours=cfg.carry_early_wave_horizon_hours,
                pick_start_hour=cfg.send_start_hour,
                due_offset_hours=cfg.due_offset_hours,
                top_n=cfg.carry_early_wave_rack_top_n,
                overnight_only=cfg.carry_early_wave_overnight_only,
            )

        if carry_rack_bootstrap:
            from slotting.inventory_policy import bootstrap_unslotted_to_rack

            setup_ranks = (
                {sku.item_id: sku.rank for sku in ranked_skus}
                if ranked_skus
                else {}
            )
            bootstrap_unslotted_to_rack(
                pallets,
                slots,
                rack_by_item=rack_by_item,
                setup_ranks=setup_ranks,
                as_of=state.as_of,
            )

        if setup_maximize_rack_fill:
            from slotting.inventory_policy import maximize_setup_rack_fill

            setup_ranks = (
                {sku.item_id: sku.rank for sku in ranked_skus}
                if ranked_skus
                else {}
            )
            maximize_setup_rack_fill(
                pallets,
                slots,
                setup_ranks=setup_ranks,
                rack_by_item=rack_by_item,
                as_of=state.as_of,
            )

    return WorldState(
        pallets=pallets,
        slots=slots,
        orders=state.orders,
        exclusions=state.exclusions,
        as_of=state.as_of,
    )


def _fill_setup_empty_sources(
    pallets: list[Pallet],
    slots: list[Slot],
    *,
    ranked_skus: list,
    face_by_item: dict[str, str],
    max_sources: int,
) -> None:
    """Stage reserve pallets onto empty source faces after pad-role swaps."""
    from slotting.inventory_policy import clear_pending_completion

    slot_by_id = {s.slot_id: s for s in slots}
    reserve = _reserve_pool(pallets)
    on_face = {
        p.item_id
        for p in pallets
        if p.current_slot
        and (slot := slot_by_id.get(p.current_slot)) is not None
        and slot.slot_type == SlotType.SOURCE
    }
    empty_sources = sorted(
        s.slot_id for s in slots if s.slot_type == SlotType.SOURCE and not s.occupied_by_pallet_id
    )
    if not empty_sources:
        return
    empty_set = set(empty_sources)
    staged_items = set(on_face)

    for sku in ranked_skus[:max_sources]:
        if not empty_sources:
            break
        if sku.item_id in staged_items:
            continue
        face_id = face_by_item.get(sku.item_id)
        if face_id not in empty_set:
            continue
        pallet = _take_reserve(reserve, sku.item_id)
        if pallet is None:
            continue
        slot = slot_by_id[face_id]
        clear_pending_completion(pallet)
        pallet.current_slot = face_id
        slot.occupied_by_pallet_id = pallet.pallet_id
        empty_set.discard(face_id)
        empty_sources.remove(face_id)
        staged_items.add(sku.item_id)

    for sku in ranked_skus:
        if not empty_sources:
            break
        if sku.item_id in staged_items:
            continue
        pallet = _take_reserve(reserve, sku.item_id)
        if pallet is None:
            continue
        slot_id = empty_sources.pop(0)
        slot = slot_by_id[slot_id]
        clear_pending_completion(pallet)
        pallet.current_slot = slot_id
        slot.occupied_by_pallet_id = pallet.pallet_id
        staged_items.add(sku.item_id)


def _effective_preopen_wave_count(
    orders: list[OrderLine],
    ship_date: date,
    *,
    cfg: ReplayConfig,
) -> int:
    """Extend setup pre-open to more waves on spike ship days."""
    count = cfg.preopen_wave_count
    from layout_advisor.planning.donor_build_queue import is_spike_ship_day

    if is_spike_ship_day(
        orders,
        ship_date,
        sku_threshold=cfg.rolling_preopen_spike_sku_threshold,
        layer_threshold=cfg.rolling_preopen_spike_layer_threshold,
        pick_start_hour=cfg.send_start_hour,
        final_shift_hour=cfg.final_shift_hour,
    ):
        count = max(count, cfg.spike_preopen_wave_count)
    return count


def _uses_rolling_preopen(
    orders: list[OrderLine],
    ship_date: date,
    *,
    cfg: ReplayConfig,
) -> bool:
    """Pull afternoon donor opens earlier on spike days (even if rolling flag is off)."""
    if cfg.rolling_donor_preopen_enabled:
        return True
    from layout_advisor.planning.donor_build_queue import is_spike_ship_day

    return is_spike_ship_day(
        orders,
        ship_date,
        sku_threshold=cfg.rolling_preopen_spike_sku_threshold,
        layer_threshold=cfg.rolling_preopen_spike_layer_threshold,
        pick_start_hour=cfg.send_start_hour,
        final_shift_hour=cfg.final_shift_hour,
    )


def _spike_preopen_max_builds(
    orders: list[OrderLine],
    ship_date: date,
    *,
    cfg: ReplayConfig,
    pad_count: int,
) -> int:
    """How many donor builds to stage at setup to stay ahead of peak waved demand."""
    from layout_advisor.planning.donor_build_queue import ship_day_peak_waved_demand

    peak_skus, _peak_layers = ship_day_peak_waved_demand(
        orders,
        ship_date,
        pick_start_hour=cfg.send_start_hour,
        final_shift_hour=cfg.final_shift_hour,
    )
    per_picker = max(1, int(math.ceil(peak_skus / max(1, cfg.effective_picker_count()))))
    return min(pad_count, max(cfg.preopen_wave_count, per_picker))


def _carried_face_by_item(state: WorldState | None) -> dict[str, str]:
    """Item → occupied source face from a carried floor state (stay-put pinning)."""
    if state is None:
        return {}
    out: dict[str, str] = {}
    for slot in state.source_slots:
        pid = slot.occupied_by_pallet_id
        if not pid:
            continue
        pallet = state.pallet_by_id(pid)
        if pallet is not None and pallet.layers_remaining > 0:
            out.setdefault(pallet.item_id, slot.slot_id)
    return out


def _donor_preopen_plan(
    orders: list[OrderLine],
    ship_date: date,
    *,
    cfg: ReplayConfig,
    pad_swaps: list | None = None,
    prev_face_by_item: dict[str, str] | None = None,
) -> list[tuple[str, int, str]]:
    """First-wave donor pre-opens aligned with planning (spike-aware on busy days)."""
    if not cfg.preopen_donor_enabled:
        return []
    from layout_advisor.planning.donor_build_queue import is_spike_ship_day
    from layout_advisor.planning.dynamic_slotting import build_setup_slotting, donor_preopen_candidates
    from layout_advisor.planning.pad_role_schedule import effective_donor_slot_ids

    setup_as_of = final_shift_as_of(ship_date, final_shift_start_hour=cfg.final_shift_hour)
    snap = visibility_snapshot(orders, setup_as_of)
    _, assignments = build_setup_slotting(
        orders,
        ship_date,
        source_count=cfg.source_slots,
        donor_count=cfg.donor_slots,
        visible_skus=snap.visible_skus,
        reserve_cap_ratio=cfg.reserve_cap_ratio,
        final_shift_hour=cfg.final_shift_hour,
        early_wave_hours=cfg.early_wave_hours,
        due_offset_hours=cfg.due_offset_hours,
        pick_start_hour=cfg.send_start_hour,
        wave_clustering=cfg.wave_clustering,
        order_affinity=cfg.order_affinity_clustering,
        rank_anchored_chain=cfg.effective_rank_anchored_chain(orders, ship_date),
        **cfg.build_anchored_kwargs(),
        order_anchored_window_assign=cfg.order_anchored_window_assign,
        setup_ship_day_foresight=cfg.setup_ship_day_foresight,
        setup_arrival_censor=cfg.setup_arrival_censor,
        velocity_twin_faces_top_k=cfg.effective_twin_faces_top_k(orders, ship_date),
        prev_face_by_item=prev_face_by_item,
        picker_count=cfg.picker_count,
        center_gap_ft=cfg.center_gap_ft,
        skus_path=Path(cfg.skus_path) if cfg.skus_path else None,
    )
    effective_donors: list[str] | None = None
    if pad_swaps:
        effective_donors = effective_donor_slot_ids(
            physical_source=cfg.source_slots,
            physical_donor=cfg.donor_slots,
            swaps_applied=pad_swaps,
            as_of=setup_as_of,
        )
    pad_count = len(effective_donors) if effective_donors else cfg.donor_slots
    wave_count = _effective_preopen_wave_count(orders, ship_date, cfg=cfg)
    spike = cfg.spike_ahead_preopen_enabled and is_spike_ship_day(
        orders,
        ship_date,
        sku_threshold=cfg.rolling_preopen_spike_sku_threshold,
        layer_threshold=cfg.rolling_preopen_spike_layer_threshold,
        pick_start_hour=cfg.send_start_hour,
        final_shift_hour=cfg.final_shift_hour,
    )
    if spike:
        from layout_advisor.planning.donor_build_queue import build_spike_setup_preopen

        max_builds = _spike_preopen_max_builds(orders, ship_date, cfg=cfg, pad_count=pad_count)
        return build_spike_setup_preopen(
            orders,
            ship_date,
            assignments,
            wave_count=wave_count,
            max_builds=max_builds,
            afternoon_hour=cfg.rolling_preopen_afternoon_hour,
            donor_count=cfg.donor_slots,
            source_count=cfg.source_slots,
            donor_slot_ids=effective_donors,
            setup_at=setup_as_of,
            due_offset_hours=cfg.due_offset_hours,
            picker_count=cfg.picker_count,
            center_gap_ft=cfg.center_gap_ft,
        )
    return donor_preopen_candidates(
        orders,
        ship_date,
        assignments,
        wave_count=wave_count,
        waves_per_day=cfg.waves_per_day,
        shift_start_hour=cfg.send_start_hour,
        due_offset_hours=cfg.due_offset_hours,
        donor_count=cfg.donor_slots,
        source_count=cfg.source_slots,
        donor_slot_ids=effective_donors,
        setup_at=setup_as_of,
        picker_count=cfg.picker_count,
        center_gap_ft=cfg.center_gap_ft,
    )[:pad_count]


def _apply_donor_preopen(
    state: WorldState,
    preopen: list[tuple[str, int, str]],
) -> WorldState:
    """Stage empty donor build pallets for first-wave SKUs (setup crew pre-open)."""
    from layout_advisor.simulate.donor_builds import apply_donor_preopen

    new_state, _ = apply_donor_preopen(state, preopen)
    return new_state


def _donor_sim_context(
    orders: list[OrderLine],
    ship_date: date,
    *,
    cfg: ReplayConfig,
    pad_swaps: list | None = None,
    prev_face_by_item: dict[str, str] | None = None,
) -> tuple[list, list | None, int | None]:
    """Wave opens + optional pad-turnover queue for simulation replay."""
    if not cfg.wave_colocated_donor_opens:
        return [], None, None
    from layout_advisor.planning.donor_build_queue import build_donor_build_queue, is_spike_ship_day
    from layout_advisor.planning.dynamic_slotting import build_setup_slotting
    from layout_advisor.planning.pad_role_schedule import effective_donor_slot_ids
    from slotting.wave_release import order_ship_date, split_day_by_native_due

    setup_as_of = final_shift_as_of(ship_date, final_shift_start_hour=cfg.final_shift_hour)
    snap = visibility_snapshot(orders, setup_as_of)
    _, assignments = build_setup_slotting(
        orders,
        ship_date,
        source_count=cfg.source_slots,
        donor_count=cfg.donor_slots,
        visible_skus=snap.visible_skus,
        reserve_cap_ratio=cfg.reserve_cap_ratio,
        final_shift_hour=cfg.final_shift_hour,
        early_wave_hours=cfg.early_wave_hours,
        due_offset_hours=cfg.due_offset_hours,
        pick_start_hour=cfg.send_start_hour,
        wave_clustering=cfg.wave_clustering,
        order_affinity=cfg.order_affinity_clustering,
        rank_anchored_chain=cfg.effective_rank_anchored_chain(orders, ship_date),
        **cfg.build_anchored_kwargs(),
        order_anchored_window_assign=cfg.order_anchored_window_assign,
        setup_ship_day_foresight=cfg.setup_ship_day_foresight,
        setup_arrival_censor=cfg.setup_arrival_censor,
        velocity_twin_faces_top_k=cfg.effective_twin_faces_top_k(orders, ship_date),
        prev_face_by_item=prev_face_by_item,
        picker_count=cfg.picker_count,
        center_gap_ft=cfg.center_gap_ft,
        skus_path=Path(cfg.skus_path) if cfg.skus_path else None,
    )
    effective_donors: list[str] | None = None
    if pad_swaps:
        effective_donors = effective_donor_slot_ids(
            physical_source=cfg.source_slots,
            physical_donor=cfg.donor_slots,
            swaps_applied=pad_swaps,
            as_of=setup_as_of,
        )
    preopen_wave_count = _effective_preopen_wave_count(orders, ship_date, cfg=cfg)
    preopen_items = {
        item_id
        for item_id, _, _ in _donor_preopen_plan(
            orders, ship_date, cfg=cfg, pad_swaps=pad_swaps, prev_face_by_item=prev_face_by_item
        )
    }
    day_orders = [o for o in orders if order_ship_date(o) == ship_date]
    buckets = split_day_by_native_due(
        day_orders,
        ship_date=ship_date,
        due_offset_hours=cfg.due_offset_hours,
    )
    wave_opens = _wave_colocated_donor_plan(
        orders, ship_date, cfg=cfg, pad_swaps=pad_swaps, prev_face_by_item=prev_face_by_item
    )

    turnover_queue = None
    turnover_threshold = None
    spike = (
        cfg.rolling_donor_preopen_enabled
        or cfg.spike_ahead_preopen_enabled
    ) and is_spike_ship_day(
        orders,
        ship_date,
        sku_threshold=cfg.rolling_preopen_spike_sku_threshold,
        layer_threshold=cfg.rolling_preopen_spike_layer_threshold,
        pick_start_hour=cfg.send_start_hour,
        final_shift_hour=cfg.final_shift_hour,
    )
    if spike and cfg.donor_pad_turnover_enabled:
        turnover_queue = build_donor_build_queue(
            orders,
            ship_date,
            assignments,
            buckets,
            source_count=cfg.source_slots,
            donor_count=cfg.donor_slots,
            donor_slot_ids=effective_donors,
            setup_at=setup_as_of,
            preopen_wave_count=preopen_wave_count,
            due_offset_hours=cfg.due_offset_hours,
            initial_used_items=preopen_items,
            picker_count=cfg.picker_count,
            center_gap_ft=cfg.center_gap_ft,
            allow_repeat_opens=cfg.effective_repeat_donor_opens(orders, ship_date),
        )
        turnover_threshold = cfg.rolling_preopen_hourly_layer_threshold

    return wave_opens, turnover_queue, turnover_threshold


def _order_donor_plan(
    orders: list[OrderLine],
    ship_date: date,
    setup_assignments: list,
    *,
    order_schedule_config,
) -> list:
    """Per-order build-pallet donor opens (order-ahead sim path)."""
    from layout_advisor.planning.order_schedule import build_order_donor_opens

    return build_order_donor_opens(
        orders,
        ship_date,
        setup_assignments,
        config=order_schedule_config,
    )


def _wave_colocated_donor_plan(
    orders: list[OrderLine],
    ship_date: date,
    *,
    cfg: ReplayConfig,
    pad_swaps: list | None = None,
    prev_face_by_item: dict[str, str] | None = None,
) -> list:
    """Per-wave colocated donor opens (rolling schedule on spike days)."""
    if not cfg.wave_colocated_donor_opens:
        return []
    from layout_advisor.planning.donor_build_queue import build_wave_donor_opens
    from layout_advisor.planning.dynamic_slotting import build_setup_slotting, donor_preopen_candidates
    from layout_advisor.planning.pad_role_schedule import effective_donor_slot_ids
    from slotting.wave_release import order_ship_date, split_day_by_native_due

    setup_as_of = final_shift_as_of(ship_date, final_shift_start_hour=cfg.final_shift_hour)
    snap = visibility_snapshot(orders, setup_as_of)
    _, assignments = build_setup_slotting(
        orders,
        ship_date,
        source_count=cfg.source_slots,
        donor_count=cfg.donor_slots,
        visible_skus=snap.visible_skus,
        reserve_cap_ratio=cfg.reserve_cap_ratio,
        final_shift_hour=cfg.final_shift_hour,
        early_wave_hours=cfg.early_wave_hours,
        due_offset_hours=cfg.due_offset_hours,
        pick_start_hour=cfg.send_start_hour,
        wave_clustering=cfg.wave_clustering,
        order_affinity=cfg.order_affinity_clustering,
        rank_anchored_chain=cfg.effective_rank_anchored_chain(orders, ship_date),
        **cfg.build_anchored_kwargs(),
        order_anchored_window_assign=cfg.order_anchored_window_assign,
        setup_ship_day_foresight=cfg.setup_ship_day_foresight,
        setup_arrival_censor=cfg.setup_arrival_censor,
        velocity_twin_faces_top_k=cfg.effective_twin_faces_top_k(orders, ship_date),
        prev_face_by_item=prev_face_by_item,
        picker_count=cfg.picker_count,
        center_gap_ft=cfg.center_gap_ft,
        skus_path=Path(cfg.skus_path) if cfg.skus_path else None,
    )
    effective_donors: list[str] | None = None
    if pad_swaps:
        effective_donors = effective_donor_slot_ids(
            physical_source=cfg.source_slots,
            physical_donor=cfg.donor_slots,
            swaps_applied=pad_swaps,
            as_of=setup_as_of,
        )
    preopen_wave_count = _effective_preopen_wave_count(orders, ship_date, cfg=cfg)
    preopen = donor_preopen_candidates(
        orders,
        ship_date,
        assignments,
        wave_count=preopen_wave_count,
        donor_slot_ids=effective_donors,
        setup_at=setup_as_of,
        due_offset_hours=cfg.due_offset_hours,
        picker_count=cfg.picker_count,
        center_gap_ft=cfg.center_gap_ft,
    )
    day_orders = [o for o in orders if order_ship_date(o) == ship_date]
    buckets = split_day_by_native_due(
        day_orders,
        ship_date=ship_date,
        due_offset_hours=cfg.due_offset_hours,
    )
    return build_wave_donor_opens(
        orders,
        ship_date,
        assignments,
        buckets,
        source_count=cfg.source_slots,
        donor_count=cfg.donor_slots,
        donor_slot_ids=effective_donors,
        setup_at=setup_as_of,
        preopen_wave_count=preopen_wave_count,
        due_offset_hours=cfg.due_offset_hours,
        initial_used_items={item_id for item_id, _, _ in preopen},
        rolling_preopen_enabled=_uses_rolling_preopen(orders, ship_date, cfg=cfg),
        spike_sku_threshold=cfg.rolling_preopen_spike_sku_threshold,
        spike_layer_threshold=cfg.rolling_preopen_spike_layer_threshold,
        rolling_lead_hours=cfg.rolling_preopen_lead_hours,
        rolling_afternoon_hour=cfg.rolling_preopen_afternoon_hour,
        pick_start_hour=cfg.send_start_hour,
        final_shift_hour=cfg.final_shift_hour,
        effective_donor_count=len(effective_donors) if effective_donors else cfg.donor_slots,
        picker_count=cfg.picker_count,
        center_gap_ft=cfg.center_gap_ft,
        lull_ahead_prep_enabled=cfg.lull_ahead_prep_enabled,
        lull_ahead_utilization_pct=cfg.lull_ahead_utilization_pct,
        lull_ahead_max_lookahead_hours=cfg.lull_ahead_max_lookahead_hours,
        lull_ahead_max_builds_per_hour=cfg.lull_ahead_max_builds_per_hour,
        pick_layers_per_hour=cfg.pick_layers_per_hour,
        pad_free_on_build_complete=cfg.pad_free_on_build_complete,
        allow_repeat_opens=cfg.effective_repeat_donor_opens(orders, ship_date),
        multi_pallet_slices=cfg.wave_multi_pallet_donor_opens,
        skus_path=cfg.skus_path,
    )


def _ensure_rack_slots(slots: list[Slot], cfg: ReplayConfig) -> list[Slot]:
    """Ensure the full rack grid exists when carrying floor state between days."""
    if not cfg.resupply_rack_enabled:
        return slots
    existing = {s.slot_id for s in slots}
    stations = aisle_stations_per_flank(
        source_count=cfg.source_slots,
        donor_count=cfg.donor_slots,
    )
    extra = [
        s
        for s in build_rack_slots_aligned(
            stations_per_flank=stations,
            depth_tiers=cfg.rack_depth_tiers,
        )
        if s.slot_id not in existing
    ]
    return slots + extra


def _build_initial_state(
    orders: list[OrderLine],
    *,
    as_of: datetime,
    cfg: ReplayConfig,
    ship_date=None,
    slotting_plan: SetupSlottingPlan | None = None,
    pad_swaps: list | None = None,
    carried_state: WorldState | None = None,
    staging_pick_boost: dict[str, int] | None = None,
) -> WorldState:
    visible_orders = orders_for_as_of(orders, as_of)
    today_item_ids = _setup_reserve_item_ids(
        orders,
        visible_orders,
        ship_date=ship_date,
        setup_ship_day_foresight=cfg.setup_ship_day_foresight,
        setup_arrival_censor=cfg.setup_arrival_censor,
    )
    item_ids = list(today_item_ids)
    next_ship_day: date | None = None
    also_keep_ship_dates: tuple[date, ...] = ()
    if carried_state is not None and ship_date is not None:
        from slotting.inventory_policy import next_ship_day_after

        next_ship_day = next_ship_day_after(orders, ship_date)
        if next_ship_day is not None and cfg.carry_keep_next_ship_day:
            also_keep_ship_dates = (next_ship_day,)
        if next_ship_day is not None and cfg.carry_reserve_next_ship_day:
            next_ranked = _setup_reserve_item_ids(
                orders,
                visible_orders,
                ship_date=next_ship_day,
                setup_ship_day_foresight=cfg.setup_ship_day_foresight,
                setup_arrival_censor=cfg.setup_arrival_censor,
            )
            cap = max(0, cfg.carry_next_day_reserve_top_n)
            item_ids = sorted(set(item_ids) | set(next_ranked[:cap]))
    is_spike = bool(
        ship_date is not None and cfg._is_spike_ship_day(orders, ship_date)
    )
    exclusions = load_lp_exclusions(cfg.lp_ineligible_items_path)

    if carried_state is not None:
        from slotting.inventory import build_reserve_pallets, finalize_depleted_pallets

        pallets = [p.model_copy(deep=True) for p in carried_state.pallets]
        slots = _ensure_rack_slots(
            [s.model_copy(deep=True) for s in carried_state.slots],
            cfg,
        )
        exclusions = list(carried_state.exclusions) + exclusions
        carry_state = WorldState(
            pallets=pallets,
            slots=slots,
            orders=orders,
            exclusions=exclusions,
            as_of=as_of,
        )
        pallets = finalize_depleted_pallets(
            pallets,
            slots,
            state=carry_state,
            colocated_donor_repurpose=cfg.colocated_donor_repurpose,
        )
        sunday_reset = cfg.carry_sunday_reset_enabled and (
            not cfg.carry_sunday_reset_on_spike_only or is_spike
        )
        keep_ids: set[str] | None = None
        if (
            is_spike
            and cfg.spike_reserve_rebootstrap
            and ship_date is not None
            and slotting_plan is not None
        ):
            from slotting.inventory_policy import spike_reserve_keep_item_ids

            keep_ids = spike_reserve_keep_item_ids(
                orders,
                ship_date,
                setup_rank_by_item=slotting_plan.setup_ranks,
                staging_boost=staging_pick_boost,
                late_shift_hour=cfg.late_shift_hour,
                final_shift_hour=cfg.final_shift_hour,
            )
        if sunday_reset:
            from slotting.inventory_policy import flush_carry_sunday_reset

            flush_carry_sunday_reset(pallets, slots, as_of=as_of)
        elif (
            keep_ids is not None
            and cfg.spike_release_slotted_outside_keep
        ):
            from slotting.inventory_policy import release_slotted_outside_keep_set

            release_slotted_outside_keep_set(
                pallets,
                slots,
                keep_item_ids=keep_ids,
                as_of=as_of,
            )
        if keep_ids is not None:
            from slotting.inventory_policy import drop_unslotted_carry_not_in_plan

            pallets = drop_unslotted_carry_not_in_plan(pallets, keep_ids)
        pallets_per_sku_by_item = None
        if cfg.demand_weighted_reserve and ship_date is not None:
            pallets_per_sku_by_item = cfg.demand_weighted_pallets_by_item(
                orders, today_item_ids, ship_date
            )
            if next_ship_day is not None and cfg.carry_reserve_next_ship_day:
                next_only = [i for i in item_ids if i not in today_item_ids]
                if next_only:
                    pallets_per_sku_by_item = {
                        **pallets_per_sku_by_item,
                        **cfg.demand_weighted_pallets_by_item(
                            orders, next_only, next_ship_day
                        ),
                    }
        rebootstrap = is_spike and cfg.spike_reserve_rebootstrap
        pallets = build_reserve_pallets(
            today_item_ids,
            orders,
            ship_date,
            as_of=as_of,
            default_layers=cfg.reserve_default_layers,
            pallets_per_sku=cfg.effective_reserve_per_sku,
            pallets_per_sku_by_item=pallets_per_sku_by_item,
            existing=pallets,
            retain_unplanned_carried=not rebootstrap,
            topoff_carried_to_target=cfg.carry_reserve_topoff,
        )
        if (
            next_ship_day is not None
            and cfg.carry_reserve_next_ship_day
        ):
            next_only = [i for i in item_ids if i not in today_item_ids]
            if next_only:
                next_per_sku = (
                    pallets_per_sku_by_item
                    if pallets_per_sku_by_item is not None
                    else None
                )
                pallets = build_reserve_pallets(
                    next_only,
                    orders,
                    next_ship_day,
                    as_of=as_of,
                    default_layers=cfg.reserve_default_layers,
                    pallets_per_sku=cfg.effective_reserve_per_sku,
                    pallets_per_sku_by_item=next_per_sku,
                    existing=pallets,
                    retain_unplanned_carried=True,
                    topoff_carried_to_target=False,
                )
        if cfg.prune_orphan_reserve and ship_date is not None:
            from slotting.inventory_policy import prune_orphan_reserve

            pallets = prune_orphan_reserve(
                pallets,
                orders,
                ship_date=ship_date,
                as_of=as_of,
                max_per_sku=cfg.reserve_max_per_sku,
                demand_weighted=cfg.demand_weighted_reserve,
                default_layers=cfg.reserve_default_layers,
                demand_weighted_max=cfg.effective_demand_weighted_max_pallets(orders, ship_date),
                is_spike_day=cfg._is_spike_ship_day(orders, ship_date),
                max_unslotted_layers=cfg.reserve_layer_budget,
                late_shift_hour=cfg.late_shift_hour,
                final_shift_hour=cfg.final_shift_hour,
            )
        if (
            cfg.prune_carry_stale_unslotted
            and ship_date is not None
            and slotting_plan is not None
        ):
            from slotting.inventory_policy import prune_carry_stale_unslotted

            setup_ranks = slotting_plan.setup_ranks
            pallets = prune_carry_stale_unslotted(
                pallets,
                orders,
                ship_date=ship_date,
                setup_rank_by_item=setup_ranks,
                max_rank_keep=cfg.carry_prune_max_rank(is_spike=is_spike),
                late_shift_hour=cfg.late_shift_hour,
                final_shift_hour=cfg.final_shift_hour,
                also_keep_ship_dates=also_keep_ship_dates,
            )
    elif cfg.track_pallet_inventory and ship_date is not None:
        from slotting.inventory import build_reserve_pallets

        slots = build_slots(source_count=cfg.source_slots, donor_count=cfg.donor_slots)
        if cfg.resupply_rack_enabled:
            stations = aisle_stations_per_flank(
                source_count=cfg.source_slots,
                donor_count=cfg.donor_slots,
            )
            slots = slots + build_rack_slots_aligned(
                stations_per_flank=stations,
                depth_tiers=cfg.rack_depth_tiers,
            )
        pallets_per_sku_by_item = (
            cfg.demand_weighted_pallets_by_item(orders, item_ids, ship_date)
            if cfg.demand_weighted_reserve
            else None
        )
        pallets = build_reserve_pallets(
            item_ids,
            orders,
            ship_date,
            as_of=as_of,
            default_layers=cfg.reserve_default_layers,
            pallets_per_sku=cfg.effective_reserve_per_sku,
            pallets_per_sku_by_item=pallets_per_sku_by_item,
        )
        if cfg.prune_orphan_reserve:
            from slotting.inventory_policy import prune_orphan_reserve

            pallets = prune_orphan_reserve(
                pallets,
                orders,
                ship_date=ship_date,
                as_of=as_of,
                max_per_sku=cfg.reserve_max_per_sku,
                demand_weighted=cfg.demand_weighted_reserve,
                default_layers=cfg.reserve_default_layers,
                demand_weighted_max=cfg.effective_demand_weighted_max_pallets(orders, ship_date),
                is_spike_day=cfg._is_spike_ship_day(orders, ship_date),
                max_unslotted_layers=cfg.reserve_layer_budget,
                late_shift_hour=cfg.late_shift_hour,
                final_shift_hour=cfg.final_shift_hour,
            )
    else:
        pallets, slots = build_sweep_floor(
            source_count=cfg.source_slots,
            donor_count=cfg.donor_slots,
            waved_item_ids=item_ids,
            as_of=as_of,
            default_layers=cfg.reserve_default_layers,
            reserve_pallets_per_sku=cfg.effective_reserve_per_sku,
        )
        if cfg.resupply_rack_enabled:
            stations = aisle_stations_per_flank(
                source_count=cfg.source_slots,
                donor_count=cfg.donor_slots,
            )
            slots = slots + build_rack_slots_aligned(
                stations_per_flank=stations,
                depth_tiers=cfg.rack_depth_tiers,
            )

    if staging_pick_boost and ship_date is not None:
        from slotting.inventory import append_staging_pick_boost

        pallets = append_staging_pick_boost(
            pallets,
            orders,
            ship_date,
            boost_by_item=staging_pick_boost,
            as_of=as_of,
            default_layers=cfg.reserve_default_layers,
        )

    state = WorldState(pallets=pallets, slots=slots, orders=orders, exclusions=exclusions, as_of=as_of)
    state = _apply_setup_shift(
        state,
        max_sources=cfg.setup_source_slots,
        backup_row_enabled=cfg.backup_row_enabled and not cfg.rack_only_reserve,
        stage_single_order_backups=cfg.stage_single_order_backups,
        ship_date=ship_date,
        max_backups_staged=cfg.max_backups_staged,
        resupply_rack_enabled=cfg.resupply_rack_enabled,
        resupply_rack_capacity=cfg.resupply_rack_capacity,
        rack_only_reserve=cfg.rack_only_reserve,
        setup_rank_cfg=cfg,
        slotting_plan=slotting_plan,
        setup_evict_stale_faces=cfg.setup_evict_stale_faces,
        setup_full_reslot=cfg.setup_full_reslot,
        jit_rack_staging=cfg.jit_rack_staging,
        jit_rack_horizon_hours=cfg.jit_rack_horizon_hours,
        carry_rack_bootstrap=cfg.carry_rack_bootstrap,
        setup_rack_stage_all_planned=cfg.setup_rack_stage_all_planned,
        setup_maximize_rack_fill=cfg.setup_maximize_rack_fill,
        setup_promote_rack_to_faces=cfg.setup_promote_rack_to_faces,
        carry_setup=carried_state is not None,
    )
    if pad_swaps:
        from layout_advisor.planning.pad_role_schedule import apply_pad_role_swaps

        state, _ = apply_pad_role_swaps(state, pad_swaps)
    if (
        cfg.setup_fill_empty_source_faces
        and ship_date is not None
        and slotting_plan is not None
    ):
        pallets = [p.model_copy(deep=True) for p in state.pallets]
        slots = [s.model_copy(deep=True) for s in state.slots]
        _fill_setup_empty_sources(
            pallets,
            slots,
            ranked_skus=slotting_plan.ranked_skus,
            face_by_item=slotting_plan.face_by_item,
            max_sources=cfg.setup_source_slots,
        )
        state = WorldState(
            pallets=pallets,
            slots=slots,
            orders=state.orders,
            exclusions=state.exclusions,
            as_of=state.as_of,
        )
    if ship_date is not None and not cfg.order_schedule_enabled:
        # Wave-style single-SKU preopens deadlock order mode: they fill every
        # donor pad at setup, order-level multi-SKU opens find no free pads,
        # and require_build_face blocks every order all day.
        preopen = _donor_preopen_plan(orders, ship_date, cfg=cfg, pad_swaps=pad_swaps)
        state = _apply_donor_preopen(state, preopen)
    return state


def _midday_realignment(state: WorldState, *, cfg: ReplayConfig, ship_date, slotting_plan) -> WorldState:
    """Demand-required only: evict wrong-SKU blockers, promote rack→face, fill empty planned faces."""
    return _apply_setup_shift(
        state,
        max_sources=cfg.setup_source_slots,
        backup_row_enabled=cfg.backup_row_enabled and not cfg.rack_only_reserve,
        stage_single_order_backups=cfg.stage_single_order_backups,
        ship_date=ship_date,
        max_backups_staged=cfg.max_backups_staged,
        resupply_rack_enabled=cfg.resupply_rack_enabled,
        resupply_rack_capacity=cfg.resupply_rack_capacity,
        rack_only_reserve=cfg.rack_only_reserve,
        setup_rank_cfg=cfg,
        slotting_plan=slotting_plan,
        setup_evict_stale_faces=True,
        setup_full_reslot=False,
        jit_rack_staging=False,
        carry_rack_bootstrap=False,
        setup_rack_stage_all_planned=False,
        setup_maximize_rack_fill=False,
        setup_promote_rack_to_faces=True,
    )


def _step_row(
    result: SimulationResult,
    *,
    shift_pulls: int = 0,
    ship_date: date | None = None,
    wave_lp_same_ship_date_only: bool = False,
    preferred_face_by_item: dict[str, str] | None = None,
    face_station_by_id: dict[str, int] | None = None,
    face_flank_by_id: dict[str, str] | None = None,
) -> ReplayStepRow:
    snap = visibility_snapshot(result.baseline_state.orders, result.as_of)
    ratio = result.wave_ratio.active_ratio if result.wave_ratio else "?"
    b_wave = result.baseline_wave_throughput
    ship_date_filter = ship_date if wave_lp_same_ship_date_only else None
    from slotting.throughput import compute_wave_throughput, compute_wave_throughput_bleed

    p_wave = compute_wave_throughput(
        result.projected_state,
        ship_date=ship_date_filter,
    )
    b_pct = 100 * b_wave.layer_pick_pct if b_wave.total_layer_demand else 0.0
    p_pct = 100 * p_wave.layer_pick_pct if p_wave.total_layer_demand else 0.0
    build_pct = 100 * p_wave.layer_pick_pct if p_wave.total_layer_demand else 0.0

    build_face = compute_wave_throughput(
        result.projected_state,
        require_build_face=True,
        ship_date=ship_date_filter,
    )
    if build_face.total_layer_demand:
        build_pct = 100 * build_face.layer_pick_pct

    waits = sum(1 for r in result.recommendations if r.action.value == "FLAG_WAIT")
    from slotting.inventory import floor_inventory_layers
    from slotting.inventory_policy import pending_completion_count, pending_completion_layers

    floor_layers, reserve_layers = floor_inventory_layers(result.projected_state.pallets)
    pending_n = pending_completion_count(result.projected_state.pallets)
    pending_layers = pending_completion_layers(result.projected_state.pallets)
    wave_tp = p_wave
    ship_day_pct = 0.0
    bleed_pct = 0.0
    ship_day_layers = 0
    bleed_layers = 0
    if ship_date is not None:
        ship_wave = compute_wave_throughput(
            result.projected_state,
            ship_date=ship_date,
        )
        bleed_wave = compute_wave_throughput_bleed(
            result.projected_state,
            ship_date,
        )
        ship_day_pct = round(100 * ship_wave.layer_pick_pct, 1) if ship_wave.total_layer_demand else 0.0
        bleed_pct = round(100 * bleed_wave.layer_pick_pct, 1) if bleed_wave.total_layer_demand else 0.0
        ship_day_layers = ship_wave.total_layer_demand
        bleed_layers = bleed_wave.total_layer_demand
    from layout_advisor.layout.travel_kpi import compute_wave_travel_kpi

    travel = compute_wave_travel_kpi(
        result.projected_state,
        preferred_face_by_item=preferred_face_by_item or {},
        face_station_by_id=face_station_by_id or {},
        face_flank_by_id=face_flank_by_id or {},
    )
    return ReplayStepRow(
        as_of=result.as_of,
        visible_skus=snap.visible_skus,
        waved_skus=snap.waved_skus,
        waved_layers=snap.waved_layers,
        wave_lp_baseline_pct=round(b_pct, 1),
        wave_lp_projected_pct=round(p_pct, 1),
        build_face_lp_pct=round(build_pct, 1),
        dead_slot_pct=round(result.projected.dead_slot_rate * 100, 1),
        shortage_count=result.projected.shortage_count,
        adds=result.adds_applied,
        removes=result.removes_applied,
        waits=waits,
        backup_promotes=result.backup_promotes_applied,
        rack_promotes=result.rack_promotes_applied,
        resupply_relocates=result.resupply_relocates_applied,
        donor_stages=result.donor_stages_applied,
        storage_returns=result.storage_returns_applied,
        active_ratio=ratio,
        floor_layers_remaining=floor_layers,
        reserve_layers_remaining=reserve_layers,
        pending_completion_pallets=pending_n,
        pending_completion_layers=pending_layers,
        lp_ready_layers=wave_tp.layer_pick_ready_layers,
        hand_unstaged_layers=wave_tp.hand_pick_unstaged_layers,
        hand_shortage_layers=wave_tp.hand_pick_shortage_layers,
        hand_excluded_layers=wave_tp.hand_pick_excluded_layers,
        low_layer_slotted_layers=wave_tp.low_layer_slotted_layers,
        shift_pulls=shift_pulls,
        wave_lp_ship_day_pct=ship_day_pct,
        wave_lp_bleed_pct=bleed_pct,
        waved_layers_ship_day=ship_day_layers,
        waved_layers_bleed=bleed_layers,
        lp_ready_station_span=travel.lp_ready_station_span,
        planned_station_span=travel.planned_station_span,
        flank_balance_pct=travel.flank_balance_pct,
        waved_layers_left=travel.waved_layers_left,
        waved_layers_right=travel.waved_layers_right,
    )


def _write_csv(rows: list[ReplayStepRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [f.name for f in ReplayStepRow.__dataclass_fields__.values()]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({f: getattr(row, f) if f != "as_of" else row.as_of.isoformat(sep=" ") for f in fields})


def _summarize(rows: list[ReplayStepRow], *, label: str, csv_path: Path, step_hours: float) -> ReplaySummary:
    waved = [r for r in rows if r.waved_layers > 0]
    lp_vals = [r.wave_lp_projected_pct for r in waved if r.waved_layers > 0]
    return ReplaySummary(
        period_label=label,
        step_count=len(rows),
        step_hours=step_hours,
        peak_waved_skus=max((r.waved_skus for r in rows), default=0),
        peak_waved_layers=max((r.waved_layers for r in rows), default=0),
        avg_wave_lp_projected_pct=round(sum(lp_vals) / len(lp_vals), 1) if lp_vals else 0.0,
        min_wave_lp_projected_pct=round(min(lp_vals), 1) if lp_vals else 0.0,
        avg_dead_slot_pct=round(sum(r.dead_slot_pct for r in rows) / len(rows), 1) if rows else 0.0,
        total_adds=sum(r.adds for r in rows),
        total_removes=sum(r.removes for r in rows),
        total_waits=sum(r.waits for r in rows),
        total_backup_promotes=sum(r.backup_promotes for r in rows),
        total_rack_promotes=sum(r.rack_promotes for r in rows),
        total_resupply_relocates=sum(r.resupply_relocates for r in rows),
        total_donor_stages=sum(r.donor_stages for r in rows),
        total_storage_returns=sum(r.storage_returns for r in rows),
        output_csv=csv_path,
    )


def _merge_carry_evening_boosts(
    cfg: ReplayConfig,
    orders: list[OrderLine],
    ship_date: date,
    *,
    carried_state: WorldState | None,
    hourly_staging_boosts: tuple[HourlyStagingBoost, ...],
) -> tuple[HourlyStagingBoost, ...]:
    """Append auto next-ship-day staging boost on carry days (merges with manual late lists)."""
    if not cfg.carry_evening_next_day_boost_enabled or carried_state is None:
        return hourly_staging_boosts
    from slotting.inventory_policy import build_next_ship_day_staging_boost

    next_day, auto_boost = build_next_ship_day_staging_boost(
        orders,
        ship_date,
        top_n=cfg.carry_evening_next_day_boost_top_n,
        early_wave_hours=cfg.early_wave_hours,
        pick_start_hour=cfg.send_start_hour,
        due_offset_hours=cfg.due_offset_hours,
    )
    if not auto_boost or next_day is None:
        return hourly_staging_boosts
    hour = cfg.carry_evening_next_day_hour
    merged = dict(auto_boost)
    kept: list[HourlyStagingBoost] = []
    for boost in hourly_staging_boosts:
        if boost.hour != hour:
            kept.append(boost)
            continue
        merged = {**merged, **boost.boost_by_item}
    kept.append(
        HourlyStagingBoost(
            ship_date=next_day,
            hour=hour,
            boost_by_item=merged,
            default_layers=cfg.reserve_default_layers,
            prefix="EVE",
        )
    )
    return tuple(kept)


def run_pick_day_backtest(
    orders: list[OrderLine],
    ship_date: date,
    *,
    cfg: ReplayConfig = ReplayConfig(),
    carried_state: WorldState | None = None,
    staging_pick_boost: dict[str, int] | None = None,
    hourly_staging_boosts: tuple[HourlyStagingBoost, ...] = (),
) -> PickDayReplay:
    """Full replay (dynamic pads, slotting, pulls) — shared by sim CSV, floor visual, and plans."""
    from layout_advisor.planning.dynamic_slotting import build_setup_slotting, build_setup_slotting_plan

    spread = _spread_orders(orders, cfg)
    day_orders = [o for o in spread if order_ship_date(o) == ship_date]
    if not day_orders:
        raise ValueError(f"No orders on ship date {ship_date}")

    setup_as_of = final_shift_as_of(ship_date, final_shift_start_hour=cfg.final_shift_hour)
    snap = visibility_snapshot(spread, setup_as_of)
    start = setup_as_of
    end = datetime.combine(ship_date, time(cfg.final_shift_hour))

    prev_faces = (
        _carried_face_by_item(carried_state) if cfg.setup_stay_put_faces else None
    ) or None
    slotting_plan = build_setup_slotting_plan(
        spread,
        ship_date,
        source_count=cfg.setup_source_slots,
        donor_count=cfg.donor_slots,
        final_shift_hour=cfg.final_shift_hour,
        early_wave_hours=cfg.early_wave_hours,
        due_offset_hours=cfg.due_offset_hours,
        pick_start_hour=cfg.send_start_hour,
        wave_clustering=cfg.wave_clustering,
        order_affinity=cfg.order_affinity_clustering,
        rank_anchored_chain=cfg.effective_rank_anchored_chain(spread, ship_date),
        **cfg.build_anchored_kwargs(),
        order_anchored_window_assign=cfg.order_anchored_window_assign,
        setup_ship_day_foresight=cfg.setup_ship_day_foresight,
        setup_arrival_censor=cfg.setup_arrival_censor,
        velocity_twin_faces_top_k=cfg.effective_twin_faces_top_k(spread, ship_date),
        prev_face_by_item=prev_faces,
        **cfg.picker_layout_kwargs(),
        skus_path=Path(cfg.skus_path) if cfg.skus_path else None,
    )
    pad_swaps: list = []
    if cfg.dynamic_pad_roles:
        from layout_advisor.planning.pad_role_schedule import build_pad_swaps_from_plan

        pad_swaps = build_pad_swaps_from_plan(
            slotting_plan,
            ship_date,
            source_slots=cfg.source_slots,
            donor_slots=cfg.donor_slots,
            setup_hour=cfg.final_shift_hour,
            prepeak_hour=cfg.prepeak_role_change_hour,
            orders=spread,
            pick_start_hour=cfg.send_start_hour,
            final_shift_hour=cfg.final_shift_hour,
            **cfg.pad_schedule_kwargs_for_day(spread, ship_date),
        )
    # Reuse the plan built above when both calls would derive the identical one
    # (reference-config path) — building it twice per sim day doubled slotting cost.
    reusable_plan = (
        slotting_plan
        if spread is orders and cfg.setup_source_slots == cfg.source_slots
        else None
    )
    _, assignments_list = build_setup_slotting(
        spread,
        ship_date,
        source_count=cfg.source_slots,
        donor_count=cfg.donor_slots,
        visible_skus=snap.visible_skus,
        reserve_cap_ratio=cfg.reserve_cap_ratio,
        final_shift_hour=cfg.final_shift_hour,
        early_wave_hours=cfg.early_wave_hours,
        due_offset_hours=cfg.due_offset_hours,
        pick_start_hour=cfg.send_start_hour,
        wave_clustering=cfg.wave_clustering,
        order_affinity=cfg.order_affinity_clustering,
        rank_anchored_chain=cfg.effective_rank_anchored_chain(orders, ship_date),
        **cfg.build_anchored_kwargs(),
        order_anchored_window_assign=cfg.order_anchored_window_assign,
        setup_ship_day_foresight=cfg.setup_ship_day_foresight,
        setup_arrival_censor=cfg.setup_arrival_censor,
        velocity_twin_faces_top_k=cfg.effective_twin_faces_top_k(orders, ship_date),
        prev_face_by_item=prev_faces,
        plan=reusable_plan,
        **cfg.picker_layout_kwargs(),
        skus_path=Path(cfg.skus_path) if cfg.skus_path else None,
    )
    preopen = (
        []
        if cfg.order_schedule_enabled  # order-level opens replace wave preopens
        else _donor_preopen_plan(
            spread, ship_date, cfg=cfg, pad_swaps=pad_swaps or None, prev_face_by_item=prev_faces
        )
    )
    initial = _build_initial_state(
        spread,
        as_of=setup_as_of,
        cfg=cfg,
        ship_date=ship_date,
        slotting_plan=slotting_plan,
        pad_swaps=pad_swaps or None,
        carried_state=carried_state,
        staging_pick_boost=staging_pick_boost,
    )
    _, aisle_map = build_aisle_slot_map(
        source_count=cfg.setup_source_slots,
        donor_count=cfg.donor_slots,
        **cfg.aisle_layout_kwargs(),
    )
    face_station_by_id, face_flank_by_id = source_face_geography(aisle_map)
    rule_config = cfg.rule_config.with_slotting_maps(
        preferred_faces=slotting_plan.preferred_faces,
        setup_ranks=slotting_plan.setup_ranks,
        face_station_by_id=face_station_by_id,
        face_flank_by_id=face_flank_by_id,
        rack_slot_by_item=slotting_plan.rack_by_item,
        twin_faces=slotting_plan.twin_face_by_item,
    ).with_policy_flags(
        visible_prestage=cfg.visible_prestage_enabled,
        aggressive_zero_demand_reclaim=cfg.aggressive_zero_demand_reclaim,
        zero_demand_reclaim_horizon_hours=cfg.zero_demand_reclaim_horizon_hours,
        proactive_zero_demand_remove=cfg.proactive_zero_demand_remove,
        proactive_zero_demand_realignment_only=cfg.proactive_zero_demand_realignment_only,
        proactive_zero_demand_realignment_hours=cfg.proactive_realignment_hours(),
        slot_reclaim_rest_of_day=cfg.slot_reclaim_rest_of_day,
        velocity_twin_face_reclaim=cfg.velocity_twin_face_reclaim,
        velocity_twin_reclaim_horizon_hours=cfg.velocity_twin_reclaim_horizon_hours,
        waved_displacement=cfg.waved_displacement,
        waved_displacement_rack_swap=cfg.waved_displacement_rack_swap,
        late_shift_hour=cfg.late_shift_hour,
        late_shift_resupply_threshold=cfg.late_shift_resupply_threshold,
        rack_promote_near_term_hour=cfg.rack_promote_near_term_hour,
        rack_promote_near_term_hours=cfg.rack_promote_near_term_hours,
        flank_local_rack_promote_batching=cfg.flank_local_rack_promote_batching,
        morning_rack_promote_batch_end_hour=cfg.morning_rack_promote_batch_end_hour,
        low_layer_refresh=cfg.low_layer_refresh_enabled,
        zero_demand_any_face_reclaim=cfg.zero_demand_any_face_reclaim,
        zero_demand_any_face_max_per_step=cfg.zero_demand_any_face_max_per_step,
        zero_demand_any_face_min_layers=cfg.zero_demand_any_face_min_layers,
    )
    wave_opens: list = []
    rolling_queue = None
    rolling_threshold = None
    order_donor_opens = None
    order_schedule_config = None
    if cfg.order_schedule_enabled:
        from layout_advisor.planning.order_schedule import OrderScheduleConfig

        order_schedule_config = OrderScheduleConfig(
            order_lookahead_hours=cfg.order_lookahead_hours,
            tick_minutes=cfg.sim_step_minutes,
            max_build_weight_lb=cfg.max_build_weight_lb,
            max_build_height_in=cfg.max_build_height_in,
            pad_free_on_build_complete=cfg.pad_free_on_build_complete,
            order_whole_order_shortage_wait=cfg.order_whole_order_shortage_wait,
            order_shortage_policy=cfg.order_shortage_policy,
            order_shortage_late_block_hours=cfg.order_shortage_late_block_hours,
            pick_start_hour=cfg.send_start_hour,
            final_shift_hour=cfg.final_shift_hour,
            final_shift_start_hour=22,
        )
        order_donor_opens = _order_donor_plan(
            spread,
            ship_date,
            assignments_list,
            order_schedule_config=order_schedule_config,
        )
    else:
        wave_opens, rolling_queue, rolling_threshold = _donor_sim_context(
            spread, ship_date, cfg=cfg, pad_swaps=pad_swaps or None, prev_face_by_item=prev_faces
        )
    inventory_rate = cfg.pick_layers_per_hour if cfg.track_pallet_inventory else None
    realignment_fn = None
    realignment_hours: list[int] = []
    if cfg.midday_realignment_hour:
        realignment_hours.append(cfg.midday_realignment_hour)
    if cfg.afternoon_realignment_hour:
        realignment_hours.append(cfg.afternoon_realignment_hour)
    if cfg.late_shift_realignment_hour:
        realignment_hours.append(cfg.late_shift_realignment_hour)
    if cfg.evening_realignment_hour:
        realignment_hours.append(cfg.evening_realignment_hour)
    if cfg.night_realignment_hour:
        realignment_hours.append(cfg.night_realignment_hour)
    if realignment_hours:
        realignment_fn = lambda state: _midday_realignment(  # noqa: E731
            state, cfg=cfg, ship_date=ship_date, slotting_plan=slotting_plan
        )

    rule_config_refresh_fn = None
    if cfg.wave_window_preferred_refresh:
        refresh_hours = set(cfg.wave_window_preferred_refresh_hours)

        def rule_config_refresh_fn(as_of, current_cfg):  # noqa: ANN001
            if as_of.hour not in refresh_hours:
                return None
            from layout_advisor.planning.wave_slotting import refresh_preferred_faces_for_window

            updated = refresh_preferred_faces_for_window(
                dict(current_cfg.preferred_source_face_by_item),
                orders=spread,
                ship_date=ship_date,
                as_of=as_of,
                aisle_map=aisle_map,
                face_by_item=dict(slotting_plan.face_by_item),
                horizon_hours=4.0,
                due_offset_hours=cfg.due_offset_hours,
            )
            if updated == dict(current_cfg.preferred_source_face_by_item):
                return None
            return current_cfg.with_slotting_maps(
                preferred_faces=updated,
                setup_ranks=current_cfg.setup_rank_by_item,
                face_station_by_id=current_cfg.source_face_station_by_id,
                face_flank_by_id=current_cfg.source_face_flank_by_id,
                rack_slot_by_item=current_cfg.rack_slot_by_item,
                twin_faces=current_cfg.twin_source_face_by_item,
            )
    pull_enabled = cfg.shift_pulls_enabled(
        carried_state=carried_state,
        ship_date=ship_date,
        orders=spread,
    )
    prep_day = cfg.is_spike_prep_pull_day(ship_date)
    carry_capped = (
        carried_state is not None
        and cfg.shift_pallet_pull_carry_capped
        and not cfg._is_spike_ship_day(spread, ship_date)
        and not prep_day
    )
    pull_max_hour_day = (
        cfg.spike_prep_pull_max_per_hour_for(ship_date)
        if prep_day and carried_state is not None
        else cfg.shift_pallet_pull_max_per_hour_carry
        if carry_capped
        else cfg.shift_pallet_pull_max_per_hour_day
    )
    if not cfg.carry_dynamic_evening_prep_enabled:
        hourly_staging_boosts = _merge_carry_evening_boosts(
            cfg,
            spread,
            ship_date,
            carried_state=carried_state,
            hourly_staging_boosts=hourly_staging_boosts,
        )
    dynamic_prep_hours: tuple[int, ...] = ()
    dynamic_prep_fn = None
    evening_carry_fn = None
    evening_carry_hours: tuple[int, ...] = ()
    prep_exclusions = list(initial.exclusions)

    if carried_state is not None and cfg.carry_dynamic_evening_prep_enabled:
        dynamic_prep_hours = (cfg.carry_evening_next_day_hour,)

        def dynamic_prep_fn(pallets, slots, orders, as_of):  # noqa: ANN001
            from slotting.inventory_policy import (
                apply_dynamic_evening_carry_prep,
                next_ship_day_after,
                stage_carry_early_wave_unslotted_to_rack,
                stage_items_to_rack,
            )

            ols_mode = cfg.carry_evening_ols_forecast_enabled
            foresight_mode = (
                cfg.carry_evening_next_morning_forecast_enabled and not ols_mode
            )
            pallets, boosted = apply_dynamic_evening_carry_prep(
                pallets,
                slots,
                orders,
                ship_date,
                as_of=as_of,
                top_n=cfg.carry_dynamic_evening_prep_top_n,
                default_layers=cfg.reserve_default_layers,
                late_shift_hour=cfg.late_shift_hour,
                final_shift_hour=cfg.final_shift_hour,
                exclusions=prep_exclusions,
                next_morning_forecast=foresight_mode,
                next_morning_ols_forecast=ols_mode,
                forecast_horizon_hours=cfg.carry_early_wave_horizon_hours,
                forecast_top_n=cfg.carry_evening_next_morning_forecast_top_n,
                forecast_overnight_only=cfg.carry_evening_next_morning_overnight_only,
                pick_start_hour=cfg.send_start_hour,
                due_offset_hours=cfg.due_offset_hours,
                ols_early_sku_fraction=cfg.carry_evening_ols_early_sku_fraction,
            )
            if cfg.carry_evening_next_day_rack_bootstrap:
                if boosted:
                    stage_items_to_rack(pallets, slots, boosted, as_of=as_of)
                # Foresight rack bootstrap peeks at the full next-day book —
                # skip it in OLS mode (minted SKUs already staged above).
                if foresight_mode:
                    next_day = next_ship_day_after(orders, ship_date)
                    if next_day is not None:
                        stage_carry_early_wave_unslotted_to_rack(
                            pallets,
                            slots,
                            orders,
                            next_day,
                            as_of=as_of,
                            horizon_hours=cfg.carry_early_wave_horizon_hours,
                            pick_start_hour=cfg.send_start_hour,
                            due_offset_hours=cfg.due_offset_hours,
                            top_n=cfg.carry_evening_next_morning_forecast_top_n,
                            overnight_only=cfg.carry_evening_next_morning_overnight_only,
                        )
            return pallets

    elif (
        carried_state is not None
        and cfg.carry_evening_next_day_rack_bootstrap
    ):
        evening_carry_hours = (cfg.carry_evening_next_day_hour,)

        def evening_carry_fn(pallets, slots, as_of):  # noqa: ANN001
            from slotting.inventory_policy import stage_evening_next_day_to_rack

            stage_evening_next_day_to_rack(
                pallets,
                slots,
                spread,
                ship_date,
                as_of=as_of,
                top_n=cfg.carry_evening_next_day_boost_top_n,
                early_wave_hours=cfg.early_wave_hours,
                pick_start_hour=cfg.send_start_hour,
                due_offset_hours=cfg.due_offset_hours,
            )
            return pallets, slots

    with pallet_height_limits(cfg.height_limits()):
        report = backtest_historical(
            initial,
            start=start,
            end=end,
            step=timedelta(minutes=cfg.sim_step_minutes),
            mode=BacktestMode.CHAINED,
            rule_config=rule_config,
            pad_role_swaps=pad_swaps or None,
            pad_role_restore_on_starve=cfg.pad_role_restore_on_starve,
            wave_donor_opens=wave_opens or None,
            rolling_donor_queue=rolling_queue,
            rolling_layer_threshold=rolling_threshold,
            pad_turnover_afternoon_hour=cfg.rolling_preopen_afternoon_hour,
            pad_turnover_lead_hours=cfg.donor_pad_turnover_lead_hours,
            inventory_layers_per_hour=inventory_rate,
            pick_shift_hours=cfg.pick_shift_hours,
            inventory_late_shift_retain=cfg.late_shift_retain_inventory,
            inventory_late_shift_hour=cfg.late_shift_hour,
            inventory_final_shift_hour=cfg.final_shift_hour,
            realignment_hours=realignment_hours or None,
            realignment_fn=realignment_fn,
            rule_config_refresh_fn=rule_config_refresh_fn,
            order_schedule_enabled=cfg.order_schedule_enabled,
            order_schedule_ship_date=ship_date if cfg.order_schedule_enabled else None,
            order_schedule_config=order_schedule_config,
            order_donor_opens=order_donor_opens,
            order_supply_complete_gate=cfg.order_supply_complete_gate,
            order_gate_due_override_hours=cfg.order_gate_due_override_hours,
            donor_open_retry_enabled=cfg.donor_open_retry_enabled,
            shift_pull=(
                ShiftPullConfig(
                    ship_date=ship_date,
                    orders=spread,
                    max_per_sku=cfg.shift_pallet_pull_max_per_sku,
                    max_per_sku_late_shift=cfg.shift_pallet_pull_max_per_sku_late,
                    late_shift_cap_min_layers=cfg.shift_pallet_pull_late_shift_min_layers,
                    max_per_hour_day=pull_max_hour_day,
                    max_per_hour_afternoon=cfg.shift_pallet_pull_max_per_hour_afternoon,
                    max_per_hour_evening=cfg.shift_pallet_pull_max_per_hour_evening,
                    default_layers=cfg.reserve_default_layers,
                    include_rack_aisle=cfg.resupply_rack_enabled,
                    resupply_threshold=cfg.late_shift_resupply_threshold,
                    stage_after_pull=cfg.shift_pallet_pull_stage_immediately,
                    preferred_faces=dict(slotting_plan.preferred_faces),
                    late_shift_hour=cfg.late_shift_hour,
                    final_shift_hour=cfg.final_shift_hour,
                    late_residual_rescue=cfg.late_shift_residual_rescue,
                )
                if pull_enabled
                else None
            ),
            idle_assist=(
                IdleAssistConfig(
                    ship_date=ship_date,
                    orders=spread,
                    max_per_hour=cfg.idle_assist_max_per_hour,
                    face_layer_threshold=cfg.idle_assist_face_layer_threshold,
                    default_layers=cfg.reserve_default_layers,
                    max_per_sku=cfg.shift_pallet_pull_max_per_sku,
                    prefer_rack=cfg.idle_assist_prefer_rack,
                    include_rack_aisle=cfg.resupply_rack_enabled,
                    require_unused_pull_budget=cfg.idle_assist_require_unused_pull_budget,
                    min_free_rack_slots=cfg.idle_assist_min_free_rack_slots,
                )
                if pull_enabled and cfg.idle_assist_staging_enabled
                else None
            ),
            hourly_staging_boosts=hourly_staging_boosts,
            dynamic_evening_prep_hours=dynamic_prep_hours,
            dynamic_evening_prep_fn=dynamic_prep_fn,
            evening_carry_hours=evening_carry_hours,
            evening_carry_fn=evening_carry_fn,
        )
        return PickDayReplay(
            report=report,
            initial=initial,
            setup_at=setup_as_of,
            pad_swaps=tuple(pad_swaps),
            wave_donor_opens=tuple(wave_opens or ()),
            assignments=tuple(assignments_list),
            preopen=tuple(preopen),
            visible_skus=snap.visible_skus,
            visible_layers=snap.visible_layers,
            ordered_layers=sum(o.qty_layers for o in day_orders),
            donor_pad_events=report.donor_pad_events,
            preferred_faces=dict(slotting_plan.preferred_faces),
            face_station_by_id=dict(face_station_by_id),
            face_flank_by_id=dict(face_flank_by_id),
        )


def write_pick_day_hourly_csv(
    replay: PickDayReplay,
    *,
    cfg: ReplayConfig,
    ship_date: date,
    path: Path,
) -> Path:
    """Write hourly sim CSV from an existing backtest replay (no re-sim)."""
    rows = replay_step_rows(replay, cfg=cfg, ship_date=ship_date)
    _write_csv(rows, path)
    return path


def replay_step_rows(
    replay: PickDayReplay,
    *,
    cfg: ReplayConfig,
    ship_date: date,
) -> list[ReplayStepRow]:
    report = replay.report
    travel_kw = dict(
        preferred_face_by_item=replay.preferred_faces,
        face_station_by_id=replay.face_station_by_id,
        face_flank_by_id=replay.face_flank_by_id,
    )
    with pallet_height_limits(cfg.height_limits()):
        return [
            _step_row(
                r,
                shift_pulls=report.shift_pulls_per_step[i] if i < len(report.shift_pulls_per_step) else 0,
                ship_date=ship_date,
                wave_lp_same_ship_date_only=cfg.wave_lp_same_ship_date_only,
                **travel_kw,
            )
            for i, r in enumerate(report.steps)
        ]


# Shift buckets for labor accounting (mirrors order_schedule.DEFAULT_PICK_SHIFTS;
# "overnight" absorbs 22:00-06:00, so picker hours nobody staffed stay visible).
PICK_SHIFT_BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("day", 6, 14),
    ("afternoon", 14, 18),
    ("evening", 18, 22),
)


def pick_labor_by_shift(events_per_step) -> dict[str, dict[str, int]]:
    """Picked layers + active pick hours per shift bucket from the event log.

    Any hour outside the named buckets (22:00-06:00) counts as "overnight" —
    labor no shift plan has staffed. A policy that grows this bucket is
    spending labor nobody priced."""
    out = {name: {"active_hours": 0, "layers": 0} for name, _, _ in PICK_SHIFT_BUCKETS}
    out["overnight"] = {"active_hours": 0, "layers": 0}
    hours_seen: dict[str, set] = {name: set() for name in out}
    for step_events in events_per_step:
        for ev in step_events:
            hour = ev.as_of.hour
            bucket = "overnight"
            for name, start, end in PICK_SHIFT_BUCKETS:
                if start <= hour < end:
                    bucket = name
                    break
            out[bucket]["layers"] += ev.layers
            hours_seen[bucket].add(ev.as_of.replace(minute=0, second=0, microsecond=0))
    for name, seen in hours_seen.items():
        out[name]["active_hours"] = len(seen)
    return out


# Hours with fewer waved layers than this cannot set the MATERIAL min-LP —
# otherwise one small order in a quiet hour can set a day's headline min LP to 0%.
MATERIAL_MIN_LP_LAYERS = 20


def material_min_lp(
    pairs,
    *,
    min_layers: int = MATERIAL_MIN_LP_LAYERS,
) -> float:
    """Min LP over (lp_pct, waved_layers) pairs with a materiality floor.

    Falls back to the raw min when no hour clears the floor."""
    pairs = [(lp, layers) for lp, layers in pairs if layers > 0]
    if not pairs:
        return 0.0
    material = [lp for lp, layers in pairs if layers >= min_layers]
    if material:
        return min(material)
    return min(lp for lp, _ in pairs)


def layer_weighted_avg_lp(pairs) -> float:
    """Day avg LP over (lp_pct, waved_layers) pairs, weighted by layers.

    The unweighted hourly mean is materiality-blind: a few near-empty hours at
    0% can swing the day average far more than the tiny share of the day's
    layers they hold justifies. Report both; weighted is the honest day-level
    aggregate."""
    pairs = [(lp, layers) for lp, layers in pairs if layers > 0]
    total = sum(layers for _, layers in pairs)
    if not total:
        return 0.0
    return round(sum(lp * layers for lp, layers in pairs) / total, 1)


def replay_sim_kpis(
    replay: PickDayReplay,
    *,
    cfg: ReplayConfig,
    ship_date: date,
) -> dict[str, float | int]:
    """Headline sim metrics for portal / daily_summary.json."""
    from layout_advisor.pick_throughput import PLANNING_CASES_PER_LAYER_DEFAULT

    rows = replay_step_rows(replay, cfg=cfg, ship_date=ship_date)
    summary = _summarize(
        rows,
        label=f"pick_day_{ship_date.isoformat()}",
        csv_path=Path(""),
        step_hours=1.0,
    )
    waved = [r for r in rows if r.waved_layers > 0]
    travel_waved = [r for r in waved if r.lp_ready_station_span > 0 or r.planned_station_span > 0]
    peak_span = max((r.lp_ready_station_span for r in travel_waved), default=0)
    avg_balance = (
        round(sum(r.flank_balance_pct for r in travel_waved) / len(travel_waved), 1)
        if travel_waved
        else 100.0
    )
    lp_ready_lph = [
        r.waved_layers * (r.wave_lp_projected_pct / 100.0) for r in waved
    ]
    avg_waved_lph = (
        round(sum(r.waved_layers for r in waved) / len(waved), 1) if waved else 0.0
    )
    avg_lp_ready_lph = (
        round(sum(lp_ready_lph) / len(lp_ready_lph), 1) if lp_ready_lph else 0.0
    )
    peak_lp_ready_lph = round(max(lp_ready_lph), 1) if lp_ready_lph else 0.0
    # Read the CONFIG field, not the module default — they are not the same knob.
    # Reading `PLANNING_CASES_PER_LAYER_DEFAULT` here instead made
    # `ReplayConfig.planning_cases_per_layer` silently inert: setting it
    # changed nothing, because the KPI reached past it to the module constant.
    # The reference value equals the default (10.0); the point is that the
    # knob works. See the calibration note on the constant itself.
    cpl = getattr(cfg, "planning_cases_per_layer", None) or PLANNING_CASES_PER_LAYER_DEFAULT
    # Fulfilment. Every other KPI here is a READINESS measure — "was the SKU on a
    # face when its layers waved" — and readiness cannot tell a floor that served
    # its book from one that picked almost none of it. That is exactly how a
    # fulfilment defect once survived.
    # layers_picked is the inventory-MUTATING path; do not substitute
    # layer_pick_events_per_step, which the order schedule fills from throwaway
    # copies and which can read high on a day the floor barely moved.
    #
    # Only the order-schedule path keeps that ledger. On the wave path demand is
    # re-requested every tick with nothing subtracted, so picks run far past the
    # book, and a day can draw several times its own book. Reporting that as a
    # fill percentage over 100% would plant the same defect somewhere new, so
    # fulfilment is reported as None there: not measured, rather than measured
    # wrong.
    ledgered = bool(getattr(cfg, "order_schedule_enabled", False))
    ordered = replay.ordered_layers
    picked = replay.report.layers_picked if ledgered else None
    fill = (
        round(100.0 * picked / ordered, 1)
        if ledgered and ordered and picked is not None
        else None
    )
    return {
        "layers_picked": picked,
        "layers_ordered": ordered,
        "fill_rate_pct": fill,
        "avg_lp_sim_pct": summary.avg_wave_lp_projected_pct,
        "avg_lp_sim_pct_weighted": layer_weighted_avg_lp(
            (r.wave_lp_projected_pct, r.waved_layers) for r in waved
        ),
        "min_lp_sim_pct": summary.min_wave_lp_projected_pct,
        "min_lp_sim_pct_material": material_min_lp(
            (r.wave_lp_projected_pct, r.waved_layers) for r in waved
        ),
        "peak_waved_layers": summary.peak_waved_layers,
        "avg_waved_layers_per_hour": avg_waved_lph,
        "avg_lp_ready_layers_per_hour": avg_lp_ready_lph,
        "peak_lp_ready_layers_per_hour": peak_lp_ready_lph,
        "avg_lp_ready_cases_per_hour": round(avg_lp_ready_lph * cpl, 0),
        "peak_lp_ready_cases_per_hour": round(peak_lp_ready_lph * cpl, 0),
        "total_waits": summary.total_waits,
        # sim tick length, so consumers can convert wait events -> starve-hours
        # (the headline wait metric) instead of quoting raw tick counts.
        "sim_step_minutes": getattr(cfg, "sim_step_minutes", None),
        "total_adds": summary.total_adds,
        "total_removes": summary.total_removes,
        "total_rack_promotes": summary.total_rack_promotes,
        "peak_lp_ready_station_span": peak_span,
        "avg_flank_balance_pct": avg_balance,
        "pick_labor_by_shift": pick_labor_by_shift(
            replay.report.layer_pick_events_per_step
        ),
    }


def simulate_pick_day(
    orders: list[OrderLine],
    ship_date: date,
    *,
    cfg: ReplayConfig = ReplayConfig(),
    out_dir: Path | None = None,
    write_csv: bool = True,
    carried_state: WorldState | None = None,
    staging_pick_boost: dict[str, int] | None = None,
    hourly_staging_boosts: tuple[HourlyStagingBoost, ...] = (),
) -> tuple[BacktestReport, ReplaySummary]:
    """Hourly chained replay from final-shift setup through end of pick day."""
    replay = run_pick_day_backtest(
        orders,
        ship_date,
        cfg=cfg,
        carried_state=carried_state,
        staging_pick_boost=staging_pick_boost,
        hourly_staging_boosts=hourly_staging_boosts,
    )
    report = replay.report
    rows = replay_step_rows(replay, cfg=cfg, ship_date=ship_date)
    label = f"pick_day_{ship_date.isoformat()}"
    csv_path = (out_dir or Path("data/sim_runs")) / f"{label}_hourly.csv"
    if write_csv:
        _write_csv(rows, csv_path)
    return report, _summarize(rows, label=label, csv_path=csv_path, step_hours=1.0)


def simulate_month_chained(
    orders: list[OrderLine],
    *,
    year: int,
    month: int,
    cfg: ReplayConfig = ReplayConfig(),
    out_dir: Path | None = None,
) -> tuple[list[ReplayStepRow], ReplaySummary]:
    """Run full hourly pick days in sequence; floor inventory carries overnight."""
    spread = _spread_orders(orders, cfg)
    start_date = date(year, month, 1)
    if month == 12:
        end_date = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        end_date = date(year, month + 1, 1) - timedelta(days=1)

    month_orders = _orders_in_date_range(spread, start=start_date, end=end_date)
    by_day = bucket_orders_by_ship_day(month_orders)
    rows: list[ReplayStepRow] = []
    carried: WorldState | None = None

    for day in sorted(by_day):
        report, _ = simulate_pick_day(
            orders,
            day,
            cfg=cfg,
            out_dir=out_dir,
            write_csv=False,
            carried_state=carried,
        )
        if not report.steps:
            continue
        eod = report.steps[-1]
        row = _step_row(eod)
        if report.final_state is not None:
            from dataclasses import replace

            from slotting.inventory import floor_inventory_layers

            floor_layers, reserve_layers = floor_inventory_layers(report.final_state.pallets)
            row = replace(
                row,
                floor_layers_remaining=floor_layers,
                reserve_layers_remaining=reserve_layers,
            )
        rows.append(row)
        carried = report.final_state or eod.projected_state

    label = f"month_{year}_{month:02d}_chained"
    csv_path = (out_dir or Path("data/sim_runs")) / f"{label}.csv"
    _write_csv(rows, csv_path)
    summary = _summarize(rows, label=label, csv_path=csv_path, step_hours=24.0)
    return rows, summary


def simulate_month_daily_summary(
    orders: list[OrderLine],
    *,
    year: int,
    month: int,
    cfg: ReplayConfig = ReplayConfig(),
    out_dir: Path | None = None,
) -> tuple[list[ReplayStepRow], ReplaySummary]:
    """One row per ship day — snapshot or chained inventory carryover."""
    if cfg.carry_inventory_across_days and cfg.track_pallet_inventory:
        return simulate_month_chained(
            orders, year=year, month=month, cfg=cfg, out_dir=out_dir
        )
    spread = _spread_orders(orders, cfg)
    start_date = date(year, month, 1)
    if month == 12:
        end_date = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        end_date = date(year, month + 1, 1) - timedelta(days=1)

    month_orders = _orders_in_date_range(spread, start=start_date, end=end_date)
    by_day = bucket_orders_by_ship_day(month_orders)
    rows: list[ReplayStepRow] = []

    from slotting.simulation import simulate_once

    with pallet_height_limits(cfg.height_limits()):
        for day in sorted(by_day):
            setup_as_of = final_shift_as_of(day, final_shift_start_hour=cfg.final_shift_hour)
            initial = _build_initial_state(spread, as_of=setup_as_of, cfg=cfg, ship_date=day)
            pick_start = datetime.combine(day, time(cfg.send_start_hour))
            pick_state = WorldState(
                pallets=initial.pallets,
                slots=initial.slots,
                orders=orders_for_as_of(spread, pick_start),
                exclusions=[],
                as_of=pick_start,
            )
            rows.append(_step_row(simulate_once(pick_state, rule_config=cfg.rule_config)))

    label = f"month_{year}_{month:02d}_daily"
    csv_path = (out_dir or Path("data/sim_runs")) / f"{label}.csv"
    _write_csv(rows, csv_path)
    summary = _summarize(rows, label=label, csv_path=csv_path, step_hours=24.0)
    return rows, summary
