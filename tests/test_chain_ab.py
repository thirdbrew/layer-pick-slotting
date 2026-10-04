"""Paired-comparison stats for chain arms — analytically verifiable cases only.

Every expected value here is hand-derivable, so a wrong implementation cannot be
"confirmed" by regenerating the fixture. That is the whole point: this module is
what future adopt/reject calls rest on.
"""

from __future__ import annotations

import math

import pytest

from scripts.chain_ab import (
    exact_signed_rank_p,
    hodges_lehmann,
    hodges_lehmann_ci,
    mde_paired,
    paired_summary,
    signed_rank_w_plus,
)


def test_w_plus_is_the_sum_of_positive_ranks():
    # |d| = 1,2,3 → ranks 1,2,3; the negative one (|d|=2, rank 2) drops out.
    assert signed_rank_w_plus([1.0, -2.0, 3.0]) == 4.0


def test_w_plus_uses_average_ranks_for_ties():
    # |d| = 1,1,3 → tied ranks 1,2 average to 1.5 each; 3 gets rank 3.
    assert signed_rank_w_plus([1.0, 1.0, 3.0]) == 6.0
    assert signed_rank_w_plus([1.0, -1.0, 3.0]) == 4.5


def test_exact_p_for_all_positive_n5_is_two_over_thirty_two():
    # Only 1 of 2^5 sign assignments reaches W+ = 15; two-sided doubles it.
    assert exact_signed_rank_p([1.0, 2.0, 3.0, 4.0, 5.0]) == pytest.approx(2 / 32)


def test_exact_p_for_a_symmetric_sample_is_one():
    assert exact_signed_rank_p([1.0, -1.0, 2.0, -2.0]) == pytest.approx(1.0)


def test_zero_deltas_are_dropped_not_counted():
    # Standard Wilcoxon handling: a day with no change carries no evidence.
    with_zeros = exact_signed_rank_p([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    assert with_zeros == pytest.approx(exact_signed_rank_p([1.0, 2.0, 3.0, 4.0, 5.0]))


def test_all_zero_deltas_give_no_evidence():
    assert exact_signed_rank_p([0.0, 0.0, 0.0]) == pytest.approx(1.0)


def test_hodges_lehmann_is_the_median_walsh_average():
    # Walsh averages of [1,2,3]: 1, 1.5, 2, 2, 2.5, 3 → median 2.0.
    assert hodges_lehmann([1.0, 2.0, 3.0]) == pytest.approx(2.0)


def test_hodges_lehmann_is_not_the_plain_median():
    # A single outlier day moves the mean a lot, the plain median not at all,
    # and HL in between — which is why the arm verdict quotes HL.
    deltas = [0.0, 0.1, 0.2, 0.3, 30.0]
    assert hodges_lehmann(deltas) > 0.2  # plain median
    assert hodges_lehmann(deltas) < sum(deltas) / len(deltas)


def test_hl_ci_brackets_the_estimate_and_widens_with_spread():
    tight = hodges_lehmann_ci([5.0, 5.1, 4.9, 5.2, 4.8, 5.05, 4.95, 5.15])
    wide = hodges_lehmann_ci([5.0, 25.0, -15.0, 5.2, -4.8, 15.05, 4.95, -5.15])
    assert tight[0] <= hodges_lehmann([5.0, 5.1, 4.9, 5.2, 4.8, 5.05, 4.95, 5.15]) <= tight[1]
    assert (wide[1] - wide[0]) > (tight[1] - tight[0])


def test_hl_ci_excludes_zero_when_every_day_improves():
    lo, hi = hodges_lehmann_ci([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    assert lo > 0.0


def test_mde_matches_the_closed_form():
    # (z_.975 + z_.80) * sd / sqrt(n) = 2.801585 * 1 / sqrt(13)
    assert mde_paired(1.0, 13) == pytest.approx(2.801585 / math.sqrt(13), rel=1e-6)


def test_mde_shrinks_with_more_days():
    assert mde_paired(1.0, 52) < mde_paired(1.0, 13)


def test_paired_summary_reports_n_after_dropping_zeros():
    out = paired_summary([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    assert out["n_days"] == 6
    assert out["n_nonzero"] == 5
    assert out["mean_delta"] == pytest.approx(15 / 6)


def test_paired_summary_flags_a_result_carried_by_few_days():
    # 2 big winners, 6 tiny losers: mean says +, the sign test says no.
    out = paired_summary([20.0, 18.0, -0.2, -0.3, -0.1, -0.4, -0.2, -0.3])
    assert out["mean_delta"] > 0
    assert out["hodges_lehmann"] < 0
    assert out["days_improved"] == 2
