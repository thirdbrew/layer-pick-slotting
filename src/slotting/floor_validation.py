"""Compare sim shortage decomposition to floor miss/short logs."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from layout_advisor.intake.columns import resolve_columns
from slotting.shortage_analysis import ShortageKind, SkuShortageRow


FLOOR_MISS_ALIASES: dict[str, list[str]] = {
    "item_id": ["item_id", "prtnum", "sku", "item"],
    "ship_date": ["ship_date", "late_shpdte", "due_date", "ship_dt"],
    "hour": ["hour", "pick_hour", "miss_hour"],
    "timestamp": ["timestamp", "as_of", "event_time", "miss_time", "recorded_at"],
    "miss_layers": [
        "miss_layers",
        "shortage_layers",
        "layers_missed",
        "hand_shortage_layers",
        "wait_layers",
        "qty_layers",
    ],
    "miss_type": ["miss_type", "shortage_type", "reason", "pick_method"],
}


@dataclass(frozen=True)
class FloorMissRow:
    item_id: str
    ship_date: date | None
    hour: int | None
    miss_layers: int
    miss_type: str | None = None


@dataclass(frozen=True)
class SkuOverlapMetrics:
    top_n: int
    sim_skus: list[str]
    floor_skus: list[str]
    overlap_skus: list[str]
    overlap_count: int
    precision_pct: float
    recall_pct: float
    jaccard_pct: float


def load_floor_miss_log(path: Path, *, ship_date: date | None = None) -> list[FloorMissRow]:
    import pandas as pd

    df = pd.read_csv(path, dtype=str)
    cols = resolve_columns(df, FLOOR_MISS_ALIASES)
    if "item_id" not in cols or "miss_layers" not in cols:
        raise ValueError(
            f"{path}: need item_id and miss_layers columns "
            f"(aliases: {FLOOR_MISS_ALIASES['item_id']}, {FLOOR_MISS_ALIASES['miss_layers']})"
        )

    rows: list[FloorMissRow] = []
    for _, raw in df.iterrows():
        item_id = str(raw[cols["item_id"]]).strip()
        if not item_id:
            continue
        miss_text = str(raw[cols["miss_layers"]]).strip()
        if not miss_text:
            continue
        miss_layers = int(float(miss_text))
        if miss_layers <= 0:
            continue

        row_ship: date | None = None
        if "ship_date" in cols and str(raw[cols["ship_date"]]).strip():
            row_ship = pd.to_datetime(raw[cols["ship_date"]]).date()
            if ship_date and row_ship != ship_date:
                continue

        hour: int | None = None
        if "hour" in cols and str(raw[cols["hour"]]).strip():
            hour = int(float(raw[cols["hour"]]))
        elif "timestamp" in cols and str(raw[cols["timestamp"]]).strip():
            hour = pd.to_datetime(raw[cols["timestamp"]]).hour

        miss_type = None
        if "miss_type" in cols and str(raw[cols["miss_type"]]).strip():
            miss_type = str(raw[cols["miss_type"]]).strip().lower()

        rows.append(
            FloorMissRow(
                item_id=item_id,
                ship_date=row_ship,
                hour=hour,
                miss_layers=miss_layers,
                miss_type=miss_type,
            )
        )
    return rows


def aggregate_floor_by_sku(
    rows: list[FloorMissRow],
    *,
    hours: tuple[int, ...] | None = None,
    miss_types: set[str] | None = None,
) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        if hours is not None and row.hour is not None and row.hour not in hours:
            continue
        if miss_types is not None and row.miss_type is not None and row.miss_type not in miss_types:
            continue
        out[row.item_id] = out.get(row.item_id, 0) + row.miss_layers
    return out


def sim_floor_pick_miss_by_sku(
    sim_rows: list[SkuShortageRow],
    *,
    include_true_shortage: bool = False,
) -> dict[str, int]:
    from slotting.shortage_analysis import FLOOR_PICK_MISS_KINDS, ShortageKind

    kinds = set(FLOOR_PICK_MISS_KINDS)
    if include_true_shortage:
        kinds.add(ShortageKind.TRUE_SHORTAGE)
    return sim_shortage_by_sku(sim_rows, kinds=kinds)


def sim_shortage_by_sku(
    sim_rows: list[SkuShortageRow],
    *,
    kinds: set[ShortageKind] | None = None,
) -> dict[str, int]:
    if kinds is None:
        kinds = {ShortageKind.TRUE_SHORTAGE}
    out: dict[str, int] = {}
    for row in sim_rows:
        if row.kind not in kinds:
            continue
        out[row.item_id] = out.get(row.item_id, 0) + row.shortage_layers
    return out


def sim_waved_by_sku(sim_rows: list[SkuShortageRow]) -> dict[str, int]:
    return {row.item_id: row.waved_layers for row in sim_rows}


def top_skus(ranked: dict[str, int], n: int) -> list[str]:
    return [item_id for item_id, _ in sorted(ranked.items(), key=lambda kv: -kv[1])[:n]]


def overlap_metrics(
    sim_ranked: dict[str, int],
    floor_ranked: dict[str, int],
    *,
    top_n: int = 10,
) -> SkuOverlapMetrics:
    sim_top = top_skus(sim_ranked, top_n)
    floor_top = top_skus(floor_ranked, top_n)
    overlap = sorted(set(sim_top) & set(floor_top))
    overlap_count = len(overlap)
    precision = 100 * overlap_count / len(sim_top) if sim_top else 0.0
    recall = 100 * overlap_count / len(floor_top) if floor_top else 0.0
    union = set(sim_top) | set(floor_top)
    jaccard = 100 * overlap_count / len(union) if union else 0.0
    return SkuOverlapMetrics(
        top_n=top_n,
        sim_skus=sim_top,
        floor_skus=floor_top,
        overlap_skus=overlap,
        overlap_count=overlap_count,
        precision_pct=round(precision, 1),
        recall_pct=round(recall, 1),
        jaccard_pct=round(jaccard, 1),
    )


def compare_sku_maps(
    sim_ranked: dict[str, int],
    floor_ranked: dict[str, int],
    *,
    top_n: int = 25,
) -> list[dict[str, object]]:
    keys = set(sim_ranked) | set(floor_ranked)
    rows: list[dict[str, object]] = []
    for item_id in keys:
        sim_layers = sim_ranked.get(item_id, 0)
        floor_layers = floor_ranked.get(item_id, 0)
        if sim_layers == 0 and floor_layers == 0:
            continue
        rows.append(
            {
                "item_id": item_id,
                "sim_layers": sim_layers,
                "floor_layers": floor_layers,
                "delta_layers": sim_layers - floor_layers,
                "in_sim_top": item_id in top_skus(sim_ranked, top_n),
                "in_floor_top": item_id in top_skus(floor_ranked, top_n),
            }
        )
    rows.sort(key=lambda r: -(int(r["sim_layers"]) + int(r["floor_layers"])))
    return rows[:top_n]


def write_comparison_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_validation_report(
    *,
    ship_date: date,
    validation_hour: int,
    sim_summary: dict,
    sim_shortage_skus: dict[str, int],
    floor_label: str,
    floor_shortage_skus: dict[str, int],
    overlap: SkuOverlapMetrics,
    comparison_rows: list[dict[str, object]],
    notes: list[str],
) -> dict:
    sim_total = sum(sim_shortage_skus.values())
    floor_total = sum(floor_shortage_skus.values())
    return {
        "ship_date": ship_date.isoformat(),
        "validation_hour": validation_hour,
        "sim": {
            "summary": sim_summary,
            "shortage_sku_count": len(sim_shortage_skus),
            "shortage_layers": sim_total,
            "top_skus": overlap.sim_skus,
        },
        "floor": {
            "source": floor_label,
            "shortage_sku_count": len(floor_shortage_skus),
            "shortage_layers": floor_total,
            "top_skus": overlap.floor_skus,
        },
        "overlap_top10": {
            "overlap_skus": overlap.overlap_skus,
            "overlap_count": overlap.overlap_count,
            "precision_pct": overlap.precision_pct,
            "recall_pct": overlap.recall_pct,
            "jaccard_pct": overlap.jaccard_pct,
            "gate_target_pct": 70.0,
            "gate_pass": overlap.recall_pct >= 70.0,
        },
        "comparison_rows": comparison_rows,
        "notes": notes,
    }
