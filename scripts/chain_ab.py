"""Paired per-day comparison of two chain arms — is the delta consistent, or two lucky days?

The chain benchmark is a mean over a handful of ship days. Comparing two means
with nothing said about spread cannot tell a +0.8pp arm from a +0.8pp reshuffle
of which days happened to win. This script puts a number on that.

**What the p-value means here, precisely.** The sim is deterministic: rerun the
same config and you get the same day. There is no Monte-Carlo sampling error to
estimate. What varies is the *day*, and the test treats the chain's ship days as
the sample. So the question answered is:

    "Does this arm beat baseline consistently across ship days, or is the mean
     carried by a couple of days that could as easily have gone the other way?"

That is the decision-relevant question for a policy that has to hold up on days
the chain has not seen. It is **not** a claim about measurement noise, and it
does not license "statistically significant improvement in LP%" to a
reader — the population being generalized to is *ship days like these*, from
one facility over one short window.

Test: two-sided Wilcoxon signed-rank, computed **exactly** by enumerating all
2^n sign assignments (n<=20). No normal approximation, no scipy dependency.
Point estimate: Hodges-Lehmann (median Walsh average), the estimator that goes
with signed-rank, plus its distribution-free CI. The mean is printed too, but
the verdict quotes HL because one 20pp day should not decide a policy.

Usage:
    # Arm files written by an A/B script (rows with base_/arm_ pairs)
    python scripts/chain_ab.py --arm-file data/sim_runs/my_arm_chain.json

    # Two chain-benchmark payloads: {"days": [{ship_date, avg_lp_sim_pct_weighted}]}
    python scripts/chain_ab.py --baseline before.json --arm after.json

    # Just the detection floor for the current chain, given an assumed spread
    python scripts/chain_ab.py --mde-only --sd 3.0 --n 5
"""

from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Two-sided alpha=0.05 and 80% power. Hardcoded rather than pulled from scipy:
# these two constants are the whole of the normal machinery this script needs.
Z_ALPHA_TWO_SIDED_95 = 1.959964
Z_POWER_80 = 0.841621

# Above this, exact enumeration (2^n) stops being free. 20 -> ~1M sign vectors,
# about a second. A full month would be ~30 days and would need the normal
# approximation instead.
# Known limitation: exact up to 20, raise instead of silently approximating. If
# a longer chain is needed, add the normal-approximation branch then.
MAX_EXACT_N = 20


def _ranks_of_absolutes(deltas: list[float]) -> list[float]:
    """Average ranks of |d|, ties shared — the standard signed-rank treatment."""
    order = sorted(range(len(deltas)), key=lambda i: abs(deltas[i]))
    ranks = [0.0] * len(deltas)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and abs(deltas[order[j + 1]]) == abs(deltas[order[i]]):
            j += 1
        shared = (i + j) / 2 + 1  # 1-based average rank of the tied block
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def _nonzero(deltas: list[float]) -> list[float]:
    """Drop zero deltas — a day that did not move carries no evidence either way."""
    return [d for d in deltas if d != 0.0]


def signed_rank_w_plus(deltas: list[float]) -> float:
    """W+ — sum of the |d| ranks belonging to positive deltas."""
    kept = _nonzero(list(deltas))
    if not kept:
        return 0.0
    ranks = _ranks_of_absolutes(kept)
    return sum(r for d, r in zip(kept, ranks) if d > 0)


def _null_w_plus_distribution(ranks: list[float]) -> list[float]:
    """Every W+ reachable by flipping signs — the exact null, all 2^n equally likely."""
    return [
        sum(r for r, s in zip(ranks, signs) if s)
        for signs in product((False, True), repeat=len(ranks))
    ]


def exact_signed_rank_p(deltas: list[float]) -> float:
    """Two-sided exact Wilcoxon signed-rank p-value. 1.0 when nothing moved."""
    kept = _nonzero(list(deltas))
    if not kept:
        return 1.0
    if len(kept) > MAX_EXACT_N:
        raise ValueError(
            f"{len(kept)} non-zero days exceeds exact enumeration limit {MAX_EXACT_N}; "
            "add the normal approximation before running a chain this long."
        )
    ranks = _ranks_of_absolutes(kept)
    observed = sum(r for d, r in zip(kept, ranks) if d > 0)
    null = _null_w_plus_distribution(ranks)
    total = len(null)
    upper = sum(1 for w in null if w >= observed) / total
    lower = sum(1 for w in null if w <= observed) / total
    return min(1.0, 2 * min(upper, lower))


def _walsh_averages(deltas: list[float]) -> list[float]:
    """(d_i + d_j)/2 for all i <= j — what HL takes the median of."""
    return sorted(
        (deltas[i] + deltas[j]) / 2 for i in range(len(deltas)) for j in range(i, len(deltas))
    )


def hodges_lehmann(deltas: list[float]) -> float:
    """Median of the Walsh averages — the location estimate signed-rank actually tests."""
    kept = _nonzero(list(deltas))
    if not kept:
        return 0.0
    walsh = _walsh_averages(kept)
    mid = len(walsh) // 2
    return walsh[mid] if len(walsh) % 2 else (walsh[mid - 1] + walsh[mid]) / 2


def hodges_lehmann_ci(deltas: list[float], alpha: float = 0.05) -> tuple[float, float]:
    """Distribution-free CI for the HL shift, cut from the exact null of W+."""
    kept = _nonzero(list(deltas))
    if not kept:
        return (0.0, 0.0)
    if len(kept) > MAX_EXACT_N:
        raise ValueError(f"{len(kept)} days exceeds exact enumeration limit {MAX_EXACT_N}")
    walsh = _walsh_averages(kept)
    ranks = _ranks_of_absolutes(kept)
    null = sorted(_null_w_plus_distribution(ranks))
    total = len(null)
    # Largest c with P(W+ <= c) <= alpha/2; trim that many Walsh averages off
    # each end. c = 0 means no trim, i.e. the widest possible interval.
    cut = 0
    for c in range(len(walsh)):
        if sum(1 for w in null if w <= c) / total > alpha / 2:
            break
        cut = c + 1
    cut = min(cut, (len(walsh) - 1) // 2)
    return (walsh[cut], walsh[len(walsh) - 1 - cut])


def mde_paired(sd_of_deltas: float, n: int, *, alpha: float = 0.05, power: float = 0.80) -> float:
    """Smallest per-day mean delta detectable at `power`, paired, normal approximation.

    Deliberately the *t*-test formula, not signed-rank: it is the standard
    reference point, it needs no tables, and it is mildly optimistic — the
    signed-rank test needs roughly 5% more effect for the same power. Quote it
    as a floor, not a promise.
    """
    if n <= 1:
        return float("inf")
    if alpha != 0.05 or power != 0.80:
        raise ValueError("only alpha=0.05 / power=0.80 are tabulated here")
    return (Z_ALPHA_TWO_SIDED_95 + Z_POWER_80) * sd_of_deltas / (n**0.5)


def _sd(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def paired_summary(deltas: list[float], *, alpha: float = 0.05) -> dict:
    """Everything an adopt/reject call needs from a list of per-day deltas."""
    deltas = [float(d) for d in deltas]
    kept = _nonzero(deltas)
    sd = _sd(deltas)
    lo, hi = hodges_lehmann_ci(deltas, alpha=alpha)
    return {
        "n_days": len(deltas),
        "n_nonzero": len(kept),
        "mean_delta": sum(deltas) / len(deltas) if deltas else 0.0,
        "sd_delta": sd,
        "hodges_lehmann": hodges_lehmann(deltas),
        "ci_low": lo,
        "ci_high": hi,
        "p_value": exact_signed_rank_p(deltas),
        "days_improved": sum(1 for d in deltas if d > 0),
        "days_worsened": sum(1 for d in deltas if d < 0),
        "worst_day_delta": min(deltas) if deltas else 0.0,
        "best_day_delta": max(deltas) if deltas else 0.0,
        "mde_at_80pct_power": mde_paired(sd, len(deltas)) if len(deltas) > 1 else float("inf"),
    }


# --------------------------------------------------------------------------
# Loading the two shapes of arm file that actually exist on disk
# --------------------------------------------------------------------------

def load_arm_file(path: Path, metric: str = "weighted") -> tuple[list[str], list[float]]:
    """Rows already carrying both arms: ``{day, base_<m>, arm_<m>}`` per day."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("days")
    if not rows:
        raise ValueError(f"{path.name} has no 'days' rows")
    base_key, arm_key = f"base_{metric}", f"arm_{metric}"
    labels: list[str] = []
    deltas: list[float] = []
    for row in rows:
        if row.get(base_key) is None or row.get(arm_key) is None:
            continue
        labels.append(str(row.get("day") or row.get("ship_date")))
        deltas.append(float(row[arm_key]) - float(row[base_key]))
    if not deltas:
        raise ValueError(f"{path.name} has no rows with both {base_key} and {arm_key}")
    return labels, deltas


def load_two_chains(
    baseline: Path, arm: Path, metric: str = "avg_lp_sim_pct_weighted"
) -> tuple[list[str], list[float]]:
    """Two chain-benchmark payloads, paired on ship_date, excluded days dropped."""

    def rows(path: Path) -> dict[str, float]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {
            r["ship_date"]: float(r[metric])
            for r in payload.get("days", [])
            if not r.get("excluded") and r.get(metric) is not None
        }

    base_rows, arm_rows = rows(baseline), rows(arm)
    shared = sorted(set(base_rows) & set(arm_rows))
    if not shared:
        raise ValueError("no ship dates in common between the two chain files")
    dropped = (set(base_rows) ^ set(arm_rows))
    if dropped:
        print(f"! {len(dropped)} unpaired day(s) dropped: {', '.join(sorted(dropped))}")
    return shared, [arm_rows[d] - base_rows[d] for d in shared]


def print_report(labels: list[str], deltas: list[float], *, title: str) -> dict:
    stats = paired_summary(deltas)
    print(f"=== Paired chain comparison - {title} ===\n")
    print(f"{'day':12}  {'delta':>7}")
    for label, delta in zip(labels, deltas):
        print(f"{label:12}  {delta:+7.2f}")
    print(
        f"\nDays: {stats['n_days']} scored, {stats['n_nonzero']} moved  "
        f"({stats['days_improved']} better, {stats['days_worsened']} worse)"
    )
    print(f"Mean delta            {stats['mean_delta']:+.2f} pp   (sd {stats['sd_delta']:.2f})")
    print(
        f"Hodges-Lehmann shift  {stats['hodges_lehmann']:+.2f} pp   "
        f"95% CI [{stats['ci_low']:+.2f}, {stats['ci_high']:+.2f}]"
    )
    print(f"Wilcoxon signed-rank  p = {stats['p_value']:.4f}  (exact, two-sided)")
    print(
        f"Worst day {stats['worst_day_delta']:+.2f} pp    "
        f"best day {stats['best_day_delta']:+.2f} pp"
    )
    print(
        f"\nDetection floor: with this spread, {stats['n_days']} days can only "
        f"resolve a mean shift of ~{stats['mde_at_80pct_power']:.2f} pp at 80% power."
    )
    consistent = stats["ci_low"] > 0 or stats["ci_high"] < 0
    print(
        "Verdict: "
        + (
            "consistent across ship days (CI excludes 0)."
            if consistent
            else "NOT consistent across ship days - the CI spans 0, so the mean is "
            "carried by day-to-day variation, not a reliable policy effect."
        )
    )
    print(
        "\nReminder: the sim is deterministic. This tests consistency across the "
        "chain's ship days, not measurement noise, and generalizes only to ship days "
        "like these."
    )
    return stats


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Paired per-day comparison of two chain arms")
    ap.add_argument("--arm-file", type=Path, help="A/B output with base_/arm_ per day")
    ap.add_argument("--baseline", type=Path, help="chain-benchmark JSON (days[].ship_date, metric)")
    ap.add_argument("--arm", type=Path, help="chain-benchmark JSON for the arm")
    ap.add_argument("--metric", default="weighted", help="arm-file metric suffix (default weighted)")
    ap.add_argument("--mde-only", action="store_true", help="just print the detection floor")
    ap.add_argument("--sd", type=float, default=3.0)
    ap.add_argument("--n", type=int, default=5, help="ship days in the chain")
    args = ap.parse_args()

    if args.mde_only:
        print(
            f"Paired MDE at n={args.n}, sd={args.sd:.2f} pp, alpha=0.05, power=0.80: "
            f"{mde_paired(args.sd, args.n):.2f} pp"
        )
        return

    if args.arm_file:
        labels, deltas = load_arm_file(args.arm_file, metric=args.metric)
        print_report(labels, deltas, title=args.arm_file.name)
    elif args.baseline and args.arm:
        labels, deltas = load_two_chains(args.baseline, args.arm)
        print_report(labels, deltas, title=f"{args.baseline.name} -> {args.arm.name}")
    else:
        ap.error("give --arm-file, or both --baseline and --arm, or --mde-only")


if __name__ == "__main__":
    main()
