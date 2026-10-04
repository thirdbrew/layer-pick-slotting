"""A reference replay configuration: a complete, self-consistent policy set.

WHAT THIS IS
------------
The simulator takes ~90 policy switches. Most combinations are incoherent -- a
reclaim rule with no reclaim budget, a prestage flag with no faces to stage
into -- so a bare `ReplayConfig()` is a poor starting point for anyone trying
the engine out. This module supplies one arrangement of those switches that is
internally consistent and produces sensible behaviour end to end.

WHAT IT IS NOT
--------------
It is not tuned, optimal, or measured. The values are a coherent starting point
and nothing more, and should be read as defaults to vary rather than as findings.

The switch NAMES and their interactions are the reusable part -- they document
what the engine can be asked to do. Sweep them with `slotting.cli simulate
--without-rules` and the A/B helpers rather than trusting any value here.
"""

from __future__ import annotations

from layout_advisor.simulate.replay import ReplayConfig

# Dynamic 70:30 physical module -- swings 65:35 (lull/setup) <-> 70:30 (peak)
# via the pad schedule. On heavy spike days a wider delta (10) gives 60:40 <-> 70:30.
REFERENCE_RACK_KWARGS = dict(
    source_slots=70,
    donor_slots=30,
    setup_source_slots=70,
    resupply_rack_enabled=True,
    rack_only_reserve=True,
    rack_depth_tiers=1,
)


REFERENCE_REPLAY_CONFIG = ReplayConfig(
    **REFERENCE_RACK_KWARGS,

    # Order-ahead dispatch: orders lock 24h before due, work starts at lock,
    # 10-minute ticks. Encodes an operating discipline rather than a KPI result.
    order_schedule_enabled=True,
    sim_step_minutes=10,

    # Inventory lifecycle
    track_pallet_inventory=True,
    carry_inventory_across_days=True,
    setup_evict_stale_faces=True,
    setup_full_reslot=True,
    carry_rack_bootstrap=True,
    prune_orphan_reserve=True,
    prioritize_pending_completion=True,
    colocated_donor_repurpose=True,

    # Dynamic pad schedule on 70:30 physical (65:35 setup/lull <-> 70:30 peak)
    dynamic_pad_roles=True,
    hourly_demand_pad_schedule=True,
    demand_triggered_pad_schedule=False,
    pad_schedule_early_source_delta=5,
    pad_schedule_spike_early_source_delta=10,

    # Staging -- faces first at setup; rack only for near-term overflow (JIT)
    jit_rack_staging=True,
    jit_rack_horizon_hours=8,
    setup_rack_stage_all_planned=False,
    setup_maximize_rack_fill=False,
    setup_promote_rack_to_faces=True,
    setup_fill_empty_source_faces=True,

    # Wave travel smoothing -- setup continuity always on; mid-day refresh opt-in
    wave_window_preferred_refresh=False,
    wave_window_preferred_refresh_hours=(11, 14),

    # Build-anchored slotting (BA) every day, with stickiness toward the carried
    # faces on light and pre-light days so a quiet day does not reshuffle the floor.
    build_anchored_slotting=True,
    setup_stay_put_faces=True,
    build_anchored_light_day_sku_threshold=30,
    build_anchored_light_day_stickiness=8.0,
    build_anchored_prelight_stickiness=True,
    wave_multi_pallet_donor_opens=False,

    # Waved-face displacement + velocity twin faces with a scaled k
    velocity_twin_faces_top_k=10,
    velocity_twin_face_reclaim=True,
    velocity_twin_reclaim_horizon_hours=4,
    velocity_twin_scaled_k_buffer=30,
    waved_displacement=True,
    demand_weighted_reserve=True,
    demand_weighted_max_pallets_per_sku=3,
    spike_day_max_pallets_per_sku=5,
    reserve_layer_budget=0,
    visible_prestage_enabled=False,
    prune_carry_stale_unslotted=True,

    # Carry-week reserve: top off thin carry, keep/mint next ship day
    carry_keep_next_ship_day=True,
    carry_reserve_next_ship_day=True,
    carry_next_day_reserve_top_n=40,
    carry_reserve_topoff=True,

    # Evening carry: shortage-driven prep @ 17:00 + rack bootstrap
    carry_evening_next_day_boost_enabled=False,
    carry_evening_next_day_hour=17,
    carry_evening_next_day_boost_top_n=30,
    carry_evening_next_day_rack_bootstrap=True,
    carry_dynamic_evening_prep_enabled=True,
    carry_dynamic_evening_prep_top_n=40,
    carry_evening_next_morning_forecast_enabled=True,
    carry_evening_next_morning_forecast_top_n=25,
    carry_evening_next_morning_overnight_only=False,
    # OLS "honesty" mint is off: full ship-day foresight remains the default.
    carry_evening_ols_forecast_enabled=False,
    carry_evening_ols_early_sku_fraction=0.25,
    carry_early_wave_rack_staging=True,
    carry_early_wave_horizon_hours=4,
    carry_early_wave_rack_top_n=25,
    carry_early_wave_overnight_only=True,  # setup: overnight only; 05–09 via evening forecast bootstrap
    rack_promote_near_term_hour=14,
    rack_promote_near_term_hours=4,
    flank_local_rack_promote_batching=True,
    # Retained for API compat; demand-first geometry tiebreak is all-day when
    # flank_local_rack_promote_batching is on (hotter SKU never delayed for path).
    morning_rack_promote_batch_end_hour=11,
    wave_lp_same_ship_date_only=False,
    wave_lp_report_ship_day_split=True,

    # No scheduled realignment sweeps — reclaim only when a waved SKU blocks a face
    midday_realignment_hour=0,
    afternoon_realignment_hour=0,
    late_shift_realignment_hour=0,
    evening_realignment_hour=0,
    night_realignment_hour=0,

    # Late-day shortage mitigation
    late_shift_hour=18,
    late_shift_retain_inventory=True,
    # 0 = any-depth sentinel (resolves to a full pallet's layers after late_shift_hour)
    late_shift_resupply_threshold=0,
    late_shift_residual_rescue=False,

    # Pick-depth model (an assumption -- set it for your clamp): the clamp takes
    # bites of at most max_lp_pick_height_in off the TOP of the stack, so a 90"
    # pallet is two bites and every layer is eventually reachable. The height
    # caps the grab, not the reachable zone.
    lp_top_pick_model=True,

    aggressive_zero_demand_reclaim=True,
    # Demand-required reclaim only — no proactive REMOVEs or face-refresh swaps
    zero_demand_reclaim_horizon_hours=4,

    # Squatter reclaim: when waved demand has no face and faces are in deficit,
    # evict the NEAREST zero-demand face rather than only the SKU's own
    # preferred face. Without it a floor can hold many stale faces while waved
    # layers have nowhere to land. Costs extra ADD+REMOVE assist moves per day.
    zero_demand_any_face_reclaim=True,
    zero_demand_any_face_max_per_step=4,
    # Minimum-layers dose gate for the eviction; 0 = off.
    zero_demand_any_face_min_layers=0,

    proactive_zero_demand_remove=False,
    proactive_zero_demand_realignment_only=False,
    proactive_zero_demand_realignment_hours=(),
    slot_reclaim_rest_of_day=True,
    low_layer_refresh_enabled=False,

    # Hybrid staging: morning prep + mid-shift pulls (fresh spike days only)
    shift_pallet_pull_enabled=True,
    shift_pallet_pull_on_carry=False,
    shift_pallet_pull_carry_capped=True,
    # Per-hour pull caps on carry days; the per-SKU cap keeps pulls from
    # polluting reserve with stock nobody needs this shift.
    shift_pallet_pull_max_per_hour_carry=8,
    shift_pallet_pull_stage_immediately=True,
    shift_pallet_pull_max_per_sku=3,
    shift_pallet_pull_max_per_sku_late=3,
    shift_pallet_pull_late_shift_min_layers=6,
    shift_pallet_pull_max_per_hour_day=8,
    shift_pallet_pull_max_per_hour_afternoon=6,
    shift_pallet_pull_max_per_hour_evening=10,
    late_shift_staging_boost_enabled=True,
    late_shift_staging_hour=17,

    # Share of outbound SKUs/layers that can be layer-picked at all; the rest
    # are hand-picked. Illustrative. Set it from your own SKU master: the
    # simulator does not read it, so it only scales reported LP downward.
    layer_pick_sku_fraction=0.75,

    # partial_line shortage policy. whole_order holds every line of an order
    # until all are supply-ready, which freezes most of the book whenever faces
    # are scarce relative to active SKUs; partial_line picks stocked lines now.
    order_shortage_policy="partial_line",
    order_whole_order_shortage_wait=False,

    # Floor validation: partial LP + top-off when face depth < waved demand
    floor_partial_pick_model_enabled=True,
    min_layers_to_layer_pick=3,
    max_layers_hand_pick=2,
    late_shift_small_pick_topoff_max_layers=6,
    max_build_height_in=100.0,
    max_lp_pick_height_in=48.0,

    # Tier 1 — spike carry transition (selective release + reserve rebootstrap; no floor flush)
    carry_sunday_reset_enabled=False,
    carry_sunday_reset_on_spike_only=True,
    spike_reserve_rebootstrap=True,
    spike_release_slotted_outside_keep=True,
    spike_carry_prune_rank_buffer=15,
    shift_pallet_pull_after_spike_reset=True,

    # Tier 2 — spike prep pulls on chosen weekdays during the carry week (capped)
    spike_prep_pull_enabled=True,
    spike_prep_pull_weekdays=(1, 2, 3, 4, 6),
    spike_prep_pull_max_per_hour=2,
    spike_prep_pull_max_per_hour_heavy=3,
    spike_prep_pull_heavy_weekdays=(3,),

    # Tier 3 — spike donor pre-open (rolling pre-open auto on spike via _uses_rolling_preopen)
    spike_ahead_preopen_enabled=True,
    spike_preopen_wave_count=8,

    rank_anchored_chain_seeding=True,
    rank_anchored_chain_spike_only=True,

    # Full ship-day book at 22:00 setup (perfect foresight). A true visibility
    # censor needs an arrived_at column in the order feed.
    setup_ship_day_foresight=True,
)
