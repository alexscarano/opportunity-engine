# -*- coding: utf-8 -*-
import sys
import os
import unittest

import numpy as np
from scipy.optimize import minimize

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
)

from elasticity_analysis import cap_channel_mix_share, optimize_with_restarts


class TestCapChannelMixShare(unittest.TestCase):

    def test_caps_dominant_channel_and_redistributes_by_historical_mix(self):
        mix = {"SKYSCANNER": 1.0, "GOOGLE": 0.0, "BING": 0.0}
        historical_mix = {"SKYSCANNER": 0.34, "GOOGLE": 0.5, "BING": 0.16}

        capped = cap_channel_mix_share(mix, historical_mix, max_share=0.4)

        self.assertAlmostEqual(capped["SKYSCANNER"], 0.4)
        self.assertAlmostEqual(sum(capped.values()), 1.0)
        # Excess (0.6) redistributed proportional to historical mix (GOOGLE 0.5 vs BING 0.16)
        self.assertGreater(capped["GOOGLE"], capped["BING"])

    def test_leaves_mix_untouched_when_no_channel_exceeds_cap(self):
        mix = {"A": 0.4, "B": 0.35, "C": 0.25}
        capped = cap_channel_mix_share(mix, historical_mix={}, max_share=0.5)
        self.assertEqual(capped, mix)

    def test_falls_back_to_even_split_when_no_signal_at_all(self):
        mix = {"A": 1.0, "B": 0.0, "C": 0.0}
        capped = cap_channel_mix_share(
            mix, historical_mix={"A": 0, "B": 0, "C": 0}, max_share=0.5
        )
        self.assertAlmostEqual(capped["A"], 0.5)
        self.assertAlmostEqual(capped["B"], 0.25)
        self.assertAlmostEqual(capped["C"], 0.25)

    def test_disabled_when_max_share_is_none_or_one(self):
        mix = {"A": 1.0, "B": 0.0}
        self.assertEqual(cap_channel_mix_share(mix, {}, None), mix)
        self.assertEqual(cap_channel_mix_share(mix, {}, 1.0), mix)


def two_basin_objective(params, _):
    """Toy non-convex objective with a shallow local minimum near x=1 (value 0.5)
    and a deeper global minimum near x=8 (value 0), used to prove multi-start
    escapes a bad local minimum a single start would get stuck in."""
    x = params[0]
    return min((x - 1) ** 2 + 0.5, (x - 8) ** 2)


class TestOptimizeWithRestarts(unittest.TestCase):

    def test_finds_better_minimum_than_a_single_bad_start(self):
        bounds = [(0.0, 10.0)]
        bad_start = [1.0]

        single_start_result = minimize(
            two_basin_objective, bad_start, args=(None,), bounds=bounds, method="L-BFGS-B"
        )
        multi_start_result = optimize_with_restarts(
            two_basin_objective, bad_start, bounds, args=(None,), n_restarts=5, seed=42
        )

        self.assertLess(multi_start_result.fun, single_start_result.fun)
        self.assertAlmostEqual(multi_start_result.fun, 0.0, places=4)

    def test_reproducible_with_a_fixed_seed(self):
        bounds = [(0.0, 10.0)]
        result_a = optimize_with_restarts(
            two_basin_objective, [1.0], bounds, args=(None,), n_restarts=5, seed=42
        )
        result_b = optimize_with_restarts(
            two_basin_objective, [1.0], bounds, args=(None,), n_restarts=5, seed=42
        )
        self.assertEqual(result_a.fun, result_b.fun)


if __name__ == "__main__":
    unittest.main()
