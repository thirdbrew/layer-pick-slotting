"""
Day-aggregate demand regression for layout-advisor Mode A sizing.

Fits ordinary least squares on ship-day history to predict:

* concurrent SKU count
* total ship-day layers
* peak wave-bucket layers

Used to size pickers and source faces for a *desired* throughput target
without treating a single historic stress day as destiny.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from collections.abc import Sequence

from slotting.models import OrderArea, OrderLine
from slotting.wave_release import order_ship_date, split_day_by_native_due

# Feature layout: intercept, weekday dummies Mon..Sat (Sun base),
# day-of-year sin/cos, lag-1 and lag-7 layers (z-scored at fit time).
_N_WEEKDAY = 6
_FEATURE_DIM = 1 + _N_WEEKDAY + 2 + 2  # intercept + Mon-Sat + season + lag1 + lag7


@dataclass(frozen=True)
class DayDemandObservation:
    """One ship day's realized demand (training row)."""

    ship_date: date
    sku_count: int
    day_layers: int
    peak_wave_layers: int
    weekday: int  # Monday=0 … Sunday=6


@dataclass(frozen=True)
class DayDemandPrediction:
    """Forecast point estimates for a planning ship date."""

    ship_date: date
    sku_count: float
    day_layers: float
    peak_wave_layers: float
    peak_layers_ratio: float
    layers_per_sku: float


@dataclass(frozen=True)
class DemandForecastModel:
    """Multi-output OLS: same feature matrix, three targets."""

    coef_sku: tuple[float, ...]
    coef_layers: tuple[float, ...]
    coef_peak: tuple[float, ...]
    lag_mean: float
    lag_std: float
    n_obs: int
    train_start: date
    train_end: date
    # Empirical percentiles from training (design safety margins)
    sku_p50: float
    sku_p90: float
    sku_p95: float
    peak_p50: float
    peak_p90: float
    peak_p95: float
    layers_p50: float
    layers_p90: float
    layers_p95: float
    r2_sku: float
    r2_layers: float
    r2_peak: float
    # Chronological holdout (last ~20%) — honest out-of-sample errors
    holdout_n: int = 0
    holdout_mae_sku: float = 0.0
    holdout_mae_layers: float = 0.0
    holdout_mae_peak: float = 0.0
    holdout_mape_peak: float = 0.0
    # Seasonal-naive (y[t-7]) on the SAME holdout rows. A raw MAE says nothing
    # about whether a model earns its complexity; a MAE next to the do-nothing
    # forecast does. Weekly seasonality is the obvious structure in ship-day
    # demand and it is what the model has to beat to be worth running.
    holdout_mae_sku_snaive: float = 0.0
    holdout_mae_layers_snaive: float = 0.0
    holdout_mae_peak_snaive: float = 0.0

    @staticmethod
    def _skill(model_mae: float, baseline_mae: float) -> float | None:
        """1 - MAE_model / MAE_baseline. Positive beats the baseline, 0 ties.

        None when the baseline is perfect (no room to score) or the holdout was
        skipped — reporting 0.0 there would read as "ties the baseline", which
        is a claim, not an absence of one.
        """
        if baseline_mae <= 0:
            return None
        return 1.0 - model_mae / baseline_mae

    @property
    def skill_sku(self) -> float | None:
        return self._skill(self.holdout_mae_sku, self.holdout_mae_sku_snaive)

    @property
    def skill_layers(self) -> float | None:
        return self._skill(self.holdout_mae_layers, self.holdout_mae_layers_snaive)

    @property
    def skill_peak(self) -> float | None:
        return self._skill(self.holdout_mae_peak, self.holdout_mae_peak_snaive)

    def predict(
        self,
        ship_date: date,
        *,
        lag_day_layers: float | None = None,
        lag7_day_layers: float | None = None,
    ) -> DayDemandPrediction:
        """Point forecast for *ship_date* (lags default to training mean)."""
        lag1 = self.lag_mean if lag_day_layers is None else float(lag_day_layers)
        lag7 = self.lag_mean if lag7_day_layers is None else float(lag7_day_layers)
        x = _feature_row(
            ship_date,
            lag1,
            lag7,
            lag_mean=self.lag_mean,
            lag_std=self.lag_std,
        )
        sku = max(1.0, _dot(self.coef_sku, x))
        layers = max(1.0, _dot(self.coef_layers, x))
        peak = max(1.0, _dot(self.coef_peak, x))
        peak = min(peak, layers)
        ratio = peak / layers
        return DayDemandPrediction(
            ship_date=ship_date,
            sku_count=sku,
            day_layers=layers,
            peak_wave_layers=peak,
            peak_layers_ratio=ratio,
            layers_per_sku=layers / sku,
        )


def build_day_observations(orders: Sequence[OrderLine]) -> list[DayDemandObservation]:
    """Aggregate THIS_LINE orders into one observation per ship day."""
    by_day: dict[date, list[OrderLine]] = defaultdict(list)
    for order in orders:
        if order.area != OrderArea.THIS_LINE:
            continue
        by_day[order_ship_date(order)].append(order)

    rows: list[DayDemandObservation] = []
    for ship_date, day_orders in sorted(by_day.items()):
        skus = {o.item_id for o in day_orders}
        day_layers = sum(o.qty_layers for o in day_orders)
        peak = _peak_wave_layers(day_orders, ship_date)
        rows.append(
            DayDemandObservation(
                ship_date=ship_date,
                sku_count=len(skus),
                day_layers=day_layers,
                peak_wave_layers=peak,
                weekday=ship_date.weekday(),
            )
        )
    return rows


def mean_rank_cum_fractions(
    orders: Sequence[OrderLine],
    *,
    max_rank: int = 200,
) -> tuple[float, ...]:
    """
    Average velocity rank curve across history.

    For each ship day, sort SKUs by layers desc and record the cumulative
    *fraction* of day layers at each rank. Average across days that have that
    many SKUs. Index ``k-1`` = mean fraction of day demand in the top-k SKUs.
    """
    by_day: dict[date, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for order in orders:
        if order.area != OrderArea.THIS_LINE:
            continue
        by_day[order_ship_date(order)][order.item_id] += order.qty_layers

    sums = [0.0] * max_rank
    counts = [0] * max_rank
    for per_sku in by_day.values():
        if not per_sku:
            continue
        total = sum(per_sku.values())
        if total <= 0:
            continue
        desc = sorted(per_sku.values(), reverse=True)
        running = 0
        for k, layers in enumerate(desc[:max_rank], start=1):
            running += layers
            sums[k - 1] += running / total
            counts[k - 1] += 1

    fracs: list[float] = []
    last = 0.0
    for i in range(max_rank):
        if counts[i] > 0:
            last = sums[i] / counts[i]
        fracs.append(min(1.0, max(last, fracs[-1] if fracs else 0.0)))
    # Ensure monotone non-decreasing and ends near 1.0 when enough ranks observed.
    for i in range(1, len(fracs)):
        fracs[i] = max(fracs[i], fracs[i - 1])
    if fracs and fracs[-1] < 1.0 and counts[0] > 0:
        # Stretch so full curve reaches 1.0 at the deepest observed average rank.
        deepest = max(i for i, c in enumerate(counts) if c > 0)
        if fracs[deepest] > 1e-9:
            scale = 1.0 / fracs[deepest]
            fracs = [min(1.0, f * scale) for f in fracs]
    return tuple(fracs)


def fit_demand_forecast(
    orders: Sequence[OrderLine],
    *,
    min_days: int = 14,
    holdout_frac: float = 0.20,
) -> DemandForecastModel:
    """
    Fit OLS day-demand model on order history.

    Coefficients are fit on the full series (for production predictions).
    Holdout MAE uses a chronological last-*holdout_frac* spit with a
    train-only re-fit so metrics are out-of-sample.

    Raises ``ValueError`` when fewer than *min_days* ship days are present.
    """
    obs = build_day_observations(orders)
    if len(obs) < min_days:
        raise ValueError(
            f"Need at least {min_days} ship days to fit demand forecast; got {len(obs)}"
        )

    lags1, lags7 = _lag_series([float(r.day_layers) for r in obs])
    lag_mean = sum(lags1) / len(lags1)
    lag_var = sum((v - lag_mean) ** 2 for v in lags1) / max(len(lags1) - 1, 1)
    lag_std = math.sqrt(lag_var) if lag_var > 1e-9 else 1.0

    x_rows = [
        _feature_row(row.ship_date, l1, l7, lag_mean=lag_mean, lag_std=lag_std)
        for row, l1, l7 in zip(obs, lags1, lags7)
    ]
    y_sku = [float(r.sku_count) for r in obs]
    y_layers = [float(r.day_layers) for r in obs]
    y_peak = [float(r.peak_wave_layers) for r in obs]

    coef_sku = _ols(x_rows, y_sku)
    coef_layers = _ols(x_rows, y_layers)
    coef_peak = _ols(x_rows, y_peak)

    holdout = _chronological_holdout(
        obs, lags1, lags7, holdout_frac=holdout_frac
    )

    return DemandForecastModel(
        coef_sku=tuple(coef_sku),
        coef_layers=tuple(coef_layers),
        coef_peak=tuple(coef_peak),
        lag_mean=lag_mean,
        lag_std=lag_std,
        n_obs=len(obs),
        train_start=obs[0].ship_date,
        train_end=obs[-1].ship_date,
        sku_p50=_percentile(y_sku, 0.50),
        sku_p90=_percentile(y_sku, 0.90),
        sku_p95=_percentile(y_sku, 0.95),
        peak_p50=_percentile(y_peak, 0.50),
        peak_p90=_percentile(y_peak, 0.90),
        peak_p95=_percentile(y_peak, 0.95),
        layers_p50=_percentile(y_layers, 0.50),
        layers_p90=_percentile(y_layers, 0.90),
        layers_p95=_percentile(y_layers, 0.95),
        r2_sku=_r2(y_sku, [_dot(coef_sku, x) for x in x_rows]),
        r2_layers=_r2(y_layers, [_dot(coef_layers, x) for x in x_rows]),
        r2_peak=_r2(y_peak, [_dot(coef_peak, x) for x in x_rows]),
        holdout_n=holdout["n"],
        holdout_mae_sku=holdout["mae_sku"],
        holdout_mae_layers=holdout["mae_layers"],
        holdout_mae_peak=holdout["mae_peak"],
        holdout_mape_peak=holdout["mape_peak"],
        holdout_mae_sku_snaive=holdout["mae_sku_snaive"],
        holdout_mae_layers_snaive=holdout["mae_layers_snaive"],
        holdout_mae_peak_snaive=holdout["mae_peak_snaive"],
    )


def predict_design_demand(
    model: DemandForecastModel,
    *,
    ship_date: date | None = None,
    safety: str = "p90",
    lag_day_layers: float | None = None,
    lag7_day_layers: float | None = None,
) -> DayDemandPrediction:
    """
    Design planning demand: point forecast blended with an empirical percentile.

    *safety* ``p50`` / ``p90`` / ``p95`` raises SKU and peak to at least that
    training percentile so footprint sizing isn't under-built on a quiet weekday
    point estimate.
    """
    when = ship_date or model.train_end
    point = model.predict(
        when,
        lag_day_layers=lag_day_layers,
        lag7_day_layers=lag7_day_layers,
    )
    sku_floor, peak_floor, layers_floor = _safety_floors(model, safety)
    sku = max(point.sku_count, sku_floor)
    peak = max(point.peak_wave_layers, peak_floor)
    layers = max(point.day_layers, layers_floor, peak)
    return DayDemandPrediction(
        ship_date=when,
        sku_count=sku,
        day_layers=layers,
        peak_wave_layers=peak,
        peak_layers_ratio=peak / layers,
        layers_per_sku=layers / sku,
    )


def _safety_floors(
    model: DemandForecastModel, safety: str
) -> tuple[float, float, float]:
    key = safety.strip().lower()
    if key == "p50":
        return model.sku_p50, model.peak_p50, model.layers_p50
    if key == "p95":
        return model.sku_p95, model.peak_p95, model.layers_p95
    if key != "p90":
        raise ValueError(f"Unknown safety level {safety!r}; use p50, p90, or p95")
    return model.sku_p90, model.peak_p90, model.layers_p90


@dataclass(frozen=True)
class QuantileDemandModel:
    """Conditional-quantile sizing model (pinball loss, IRLS fit).

    Design sizing is a newsvendor problem — under-building peak capacity costs
    far more than over-building — so estimate the tau-quantile of each target
    directly, conditioned on weekday + season, instead of a point OLS raised
    to an unconditional percentile floor. Lag features are omitted: on short
    ship-day histories their coefficients are mostly noise.
    """

    tau: float
    coef_sku: tuple[float, ...]
    coef_layers: tuple[float, ...]
    coef_peak: tuple[float, ...]
    n_obs: int
    train_start: date
    train_end: date
    # Chronological holdout — empirical coverage should sit near tau
    holdout_n: int = 0
    holdout_coverage_sku: float = 0.0
    holdout_coverage_layers: float = 0.0
    holdout_coverage_peak: float = 0.0
    holdout_mean_pred_peak: float = 0.0
    # Point accuracy on the same holdout. These existed on FittedDemandModel but
    # NOT here, and throughput_sizing read them with
    # `getattr(model, "holdout_mae_peak", 0.0)` — so a missing attribute became a
    # PERFECT score: a report could print "peak MAE 0.0 layers (0% MAPE)" for
    # a forecast with real error, while holdout_coverage_peak (which does
    # exist) looked plausible in the same block.
    holdout_mae_peak: float = 0.0
    holdout_mape_peak: float = 0.0

    def predict(self, ship_date: date) -> DayDemandPrediction:
        x = _quantile_feature_row(ship_date)
        sku = max(1.0, _dot(self.coef_sku, x))
        layers = max(1.0, _dot(self.coef_layers, x))
        peak = max(1.0, min(_dot(self.coef_peak, x), layers))
        return DayDemandPrediction(
            ship_date=ship_date,
            sku_count=sku,
            day_layers=layers,
            peak_wave_layers=peak,
            peak_layers_ratio=peak / layers,
            layers_per_sku=layers / sku,
        )


def _quantile_feature_row(ship_date: date) -> list[float]:
    """Intercept + weekday dummies + annual sin/cos (no lags)."""
    wd = ship_date.weekday()
    feats = [1.0]
    for i in range(_N_WEEKDAY):
        feats.append(1.0 if wd == i else 0.0)
    doy = ship_date.timetuple().tm_yday
    angle = 2.0 * math.pi * doy / 365.25
    feats.append(math.sin(angle))
    feats.append(math.cos(angle))
    return feats


def _quantile_fit(
    x_rows: Sequence[Sequence[float]],
    y: Sequence[float],
    *,
    tau: float,
    iterations: int = 100,
    eps: float = 1.0,
) -> list[float]:
    """Quantile regression via iteratively reweighted least squares.

    Weights w_i = |tau - 1(r_i < 0)| / max(|r_i|, eps) turn the pinball loss
    into a sequence of weighted least-squares problems. *eps* floors the
    residual at one layer — a smaller floor lets near-zero residuals dominate
    the WLS and the fit stalls at its OLS start instead of climbing to the
    quantile. Swap for an LP formulation only if coverage drifts.
    """
    coef = _ols(x_rows, y)
    p = len(x_rows[0])
    for _ in range(iterations):
        xtx = [[0.0] * p for _ in range(p)]
        xty = [0.0] * p
        for row, yi in zip(x_rows, y):
            r = yi - _dot(coef, row)
            w = (tau if r >= 0 else 1.0 - tau) / max(abs(r), eps)
            for i in range(p):
                xty[i] += w * row[i] * yi
                for j in range(p):
                    xtx[i][j] += w * row[i] * row[j]
        ridge = 1e-8 * max(1.0, sum(xtx[i][i] for i in range(p)) / p)
        for i in range(p):
            xtx[i][i] += ridge
        new = _solve(xtx, xty)
        if max(abs(a - b) for a, b in zip(new, coef)) < 1e-7:
            coef = new
            break
        coef = new
    return coef


def tau_for_safety(safety: str, *, default: float = 0.90) -> float:
    """Map a design safety level (p50/p90/p95) to its quantile tau."""
    return {"p50": 0.50, "p90": 0.90, "p95": 0.95}.get(safety.strip().lower(), default)


def fit_demand_quantile(
    orders: Sequence[OrderLine],
    *,
    tau: float = 0.90,
    min_days: int = 14,
    holdout_frac: float = 0.20,
) -> QuantileDemandModel:
    """Fit conditional tau-quantile models for sku / layers / peak targets."""
    if not 0.0 < tau < 1.0:
        raise ValueError(f"tau must be in (0,1); got {tau}")
    obs = build_day_observations(orders)
    if len(obs) < min_days:
        raise ValueError(
            f"Need at least {min_days} ship days to fit quantile model; got {len(obs)}"
        )
    x_rows = [_quantile_feature_row(r.ship_date) for r in obs]
    y_sku = [float(r.sku_count) for r in obs]
    y_layers = [float(r.day_layers) for r in obs]
    y_peak = [float(r.peak_wave_layers) for r in obs]

    coef_sku = _quantile_fit(x_rows, y_sku, tau=tau)
    coef_layers = _quantile_fit(x_rows, y_layers, tau=tau)
    coef_peak = _quantile_fit(x_rows, y_peak, tau=tau)

    n = len(obs)
    hold_n = max(1, int(round(n * holdout_frac)))
    cov_sku = cov_layers = cov_peak = mean_peak = 0.0
    err_peak = mape_peak = 0.0
    hold_used = 0
    if n - hold_n >= 10:
        split = n - hold_n
        x_train = x_rows[:split]
        c_sku = _quantile_fit(x_train, y_sku[:split], tau=tau)
        c_layers = _quantile_fit(x_train, y_layers[:split], tau=tau)
        c_peak = _quantile_fit(x_train, y_peak[:split], tau=tau)
        for i in range(split, n):
            x = x_rows[i]
            p_sku = max(1.0, _dot(c_sku, x))
            p_layers = max(1.0, _dot(c_layers, x))
            p_peak = max(1.0, min(_dot(c_peak, x), p_layers))
            cov_sku += 1.0 if y_sku[i] <= p_sku else 0.0
            cov_layers += 1.0 if y_layers[i] <= p_layers else 0.0
            cov_peak += 1.0 if y_peak[i] <= p_peak else 0.0
            mean_peak += p_peak
            # Point error alongside coverage — same loop, same holdout. A
            # quantile model at tau=0.9 is SUPPOSED to over-predict, so coverage
            # alone cannot say whether the forecast is any good.
            err_peak += abs(p_peak - y_peak[i])
            if y_peak[i] > 0:
                mape_peak += abs(p_peak - y_peak[i]) / y_peak[i]
        hold_used = hold_n
        cov_sku /= hold_n
        cov_layers /= hold_n
        cov_peak /= hold_n
        mean_peak /= hold_n
        err_peak /= hold_n
        mape_peak /= hold_n

    return QuantileDemandModel(
        tau=tau,
        coef_sku=tuple(coef_sku),
        coef_layers=tuple(coef_layers),
        coef_peak=tuple(coef_peak),
        n_obs=n,
        train_start=obs[0].ship_date,
        train_end=obs[-1].ship_date,
        holdout_n=hold_used,
        holdout_coverage_sku=round(cov_sku, 3),
        holdout_coverage_layers=round(cov_layers, 3),
        holdout_coverage_peak=round(cov_peak, 3),
        holdout_mean_pred_peak=round(mean_peak, 1),
        holdout_mae_peak=round(err_peak, 1),
        holdout_mape_peak=round(mape_peak, 3),
    )


def _peak_wave_layers(day_orders: list[OrderLine], ship_date: date) -> int:
    if not day_orders:
        return 0
    buckets = split_day_by_native_due(
        day_orders, ship_date=ship_date, due_offset_hours=4.0
    )
    if not buckets:
        return sum(o.qty_layers for o in day_orders)
    return max(b.layer_demand for b in buckets)


def _lag_series(day_layers: Sequence[float]) -> tuple[list[float], list[float]]:
    """Build lag-1 and lag-7 series (ship-day index, not calendar days)."""
    n = len(day_layers)
    lags1: list[float] = []
    lags7: list[float] = []
    for i in range(n):
        lags1.append(float(day_layers[i - 1]) if i >= 1 else float(day_layers[0]))
        lags7.append(float(day_layers[i - 7]) if i >= 7 else float(day_layers[0]))
    return lags1, lags7


def _chronological_holdout(
    obs: Sequence[DayDemandObservation],
    lags1: Sequence[float],
    lags7: Sequence[float],
    *,
    holdout_frac: float,
) -> dict[str, float | int]:
    n = len(obs)
    hold_n = max(1, int(round(n * holdout_frac)))
    if n - hold_n < 10:
        return {
            "n": 0,
            "mae_sku": 0.0,
            "mae_layers": 0.0,
            "mae_peak": 0.0,
            "mape_peak": 0.0,
            "mae_sku_snaive": 0.0,
            "mae_layers_snaive": 0.0,
            "mae_peak_snaive": 0.0,
        }
    split = n - hold_n
    train_layers = [float(r.day_layers) for r in obs[:split]]
    lag_mean = sum(train_layers) / len(train_layers)
    lag_var = sum((v - lag_mean) ** 2 for v in train_layers) / max(len(train_layers) - 1, 1)
    lag_std = math.sqrt(lag_var) if lag_var > 1e-9 else 1.0

    x_train = [
        _feature_row(obs[i].ship_date, lags1[i], lags7[i], lag_mean=lag_mean, lag_std=lag_std)
        for i in range(split)
    ]
    coef_sku = _ols(x_train, [float(r.sku_count) for r in obs[:split]])
    coef_layers = _ols(x_train, [float(r.day_layers) for r in obs[:split]])
    coef_peak = _ols(x_train, [float(r.peak_wave_layers) for r in obs[:split]])

    err_sku = err_layers = err_peak = mape_peak = 0.0
    # Seasonal naive: "same weekday last week". Scored on the identical rows, so
    # the comparison is like-for-like. Indexing obs[i - 7] is safe because the
    # holdout starts at `split` and the guard above kept at least 10 train rows.
    sn_sku = sn_layers = sn_peak = 0.0
    for i in range(split, n):
        x = _feature_row(
            obs[i].ship_date, lags1[i], lags7[i], lag_mean=lag_mean, lag_std=lag_std
        )
        pred_sku = max(1.0, _dot(coef_sku, x))
        pred_layers = max(1.0, _dot(coef_layers, x))
        pred_peak = max(1.0, min(_dot(coef_peak, x), pred_layers))
        err_sku += abs(pred_sku - obs[i].sku_count)
        err_layers += abs(pred_layers - obs[i].day_layers)
        err_peak += abs(pred_peak - obs[i].peak_wave_layers)
        if obs[i].peak_wave_layers > 0:
            mape_peak += abs(pred_peak - obs[i].peak_wave_layers) / obs[i].peak_wave_layers
        prior = obs[i - 7]
        sn_sku += abs(prior.sku_count - obs[i].sku_count)
        sn_layers += abs(prior.day_layers - obs[i].day_layers)
        sn_peak += abs(prior.peak_wave_layers - obs[i].peak_wave_layers)

    return {
        "n": hold_n,
        "mae_sku": round(err_sku / hold_n, 2),
        "mae_layers": round(err_layers / hold_n, 1),
        "mae_peak": round(err_peak / hold_n, 1),
        "mape_peak": round(mape_peak / hold_n, 3),
        "mae_sku_snaive": round(sn_sku / hold_n, 2),
        "mae_layers_snaive": round(sn_layers / hold_n, 1),
        "mae_peak_snaive": round(sn_peak / hold_n, 1),
    }


def _feature_row(
    ship_date: date,
    lag_day_layers: float,
    lag7_day_layers: float,
    *,
    lag_mean: float,
    lag_std: float,
) -> list[float]:
    wd = ship_date.weekday()  # Mon=0
    feats = [1.0]
    for i in range(_N_WEEKDAY):
        feats.append(1.0 if wd == i else 0.0)
    doy = ship_date.timetuple().tm_yday
    angle = 2.0 * math.pi * doy / 365.25
    feats.append(math.sin(angle))
    feats.append(math.cos(angle))
    feats.append((lag_day_layers - lag_mean) / lag_std)
    feats.append((lag7_day_layers - lag_mean) / lag_std)
    assert len(feats) == _FEATURE_DIM
    return feats


def _dot(coef: Sequence[float], x: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(coef, x))


def _ols(x_rows: Sequence[Sequence[float]], y: Sequence[float]) -> list[float]:
    """Solve (X'X) β = X'y with Gaussian elimination; ridge if singular."""
    p = len(x_rows[0])
    xtx = [[0.0] * p for _ in range(p)]
    xty = [0.0] * p
    for row, yi in zip(x_rows, y):
        for i in range(p):
            xty[i] += row[i] * yi
            for j in range(p):
                xtx[i][j] += row[i] * row[j]
    ridge = 1e-6 * max(1.0, sum(xtx[i][i] for i in range(p)) / p)
    for i in range(p):
        xtx[i][i] += ridge
    return _solve(xtx, xty)


def _solve(a: list[list[float]], b: list[float]) -> list[float]:
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            continue
        m[col], m[pivot] = m[pivot], m[col]
        div = m[col][col]
        for j in range(col, n + 1):
            m[col][j] /= div
        for row in range(n):
            if row == col:
                continue
            factor = m[row][col]
            for j in range(col, n + 1):
                m[row][j] -= factor * m[col][j]
    return [m[i][n] for i in range(n)]


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    rank = q * (len(xs) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(xs) - 1)
    w = rank - lo
    return xs[lo] * (1.0 - w) + xs[hi] * w


def _r2(y: Sequence[float], yhat: Sequence[float]) -> float:
    if not y:
        return 0.0
    mean = sum(y) / len(y)
    ss_tot = sum((v - mean) ** 2 for v in y)
    if ss_tot < 1e-12:
        return 1.0
    ss_res = sum((a - b) ** 2 for a, b in zip(y, yhat))
    return max(0.0, 1.0 - ss_res / ss_tot)
